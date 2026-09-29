"""Federated self-supervised pretraining loop for the Phase 0 encoder
(Section V-C): each client trains SelfSupervisedEncoder on its own
unlabeled raw sequences via masked-packet reconstruction; the server
FedAvg's the encoder weights each round. No labels are used here at all,
consistent with the design.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from . import federated as fed
from .models import DomainDiscriminator, SelfSupervisedEncoder, gradient_reversal


def federated_pretrain_encoder(
    client_sequences: Dict[str, np.ndarray],
    seq_len: int = 100,
    num_rounds: int = 15,
    lr: float = 1e-3,
    local_epochs: int = 1,
    batch_size: int = 64,
    d_model: int = 64,
    embed_dim: int = 32,
    seed: int = 0,
    variational: bool = False,
    kl_weight: float = 0.001,
) -> SelfSupervisedEncoder:
    """`variational=True` trains a VAE-style encoder instead of the
    deterministic default -- see SelfSupervisedEncoder's docstring for
    the real prior work (Sivanathan et al.) this replicates and why."""
    torch.manual_seed(seed)
    global_encoder = SelfSupervisedEncoder(
        seq_len=seq_len, d_model=d_model, embed_dim=embed_dim, variational=variational,
    )

    client_tensors = {
        cid: torch.tensor(X, dtype=torch.float32) for cid, X in client_sequences.items()
    }

    for r in range(num_rounds):
        client_states = []
        for cid, X in client_tensors.items():
            local_model = fed.get_state(global_encoder)
            model = SelfSupervisedEncoder(
                seq_len=seq_len, d_model=d_model, embed_dim=embed_dim, variational=variational,
            )
            fed.set_state(model, local_model)
            model.train()
            opt = torch.optim.Adam(model.parameters(), lr=lr)
            for _ in range(local_epochs):
                idx = torch.randperm(len(X))
                for i in range(0, len(X), batch_size):
                    b = idx[i : i + batch_size]
                    opt.zero_grad()
                    loss = model.forward_pretrain(X[b], kl_weight=kl_weight)
                    loss.backward()
                    opt.step()
            client_states.append(fed.get_state(model))
        new_global = fed.weighted_average_states(client_states, [1.0] * len(client_states))
        fed.set_state(global_encoder, new_global)
        print(f"  Phase 0 pretraining round {r+1}/{num_rounds} done")

    return global_encoder


def federated_pretrain_encoder_domain_adversarial(
    client_sequences: Dict[str, np.ndarray],
    seq_len: int = 100,
    num_rounds: int = 15,
    lr: float = 1e-3,
    local_epochs: int = 1,
    batch_size: int = 64,
    d_model: int = 64,
    embed_dim: int = 32,
    seed: int = 0,
    grl_lambda: float = 1.0,
    domain_loss_weight: float = 1.0,
    domain_labels: "Dict[str, int] | None" = None,
) -> SelfSupervisedEncoder:
    """Domain-adversarial variant of federated_pretrain_encoder, motivated
    by a real diagnosed failure (Section VII-G): an encoder trained only
    with masked-packet reconstruction does not reliably recognize the
    same device type captured on a different physical network (1 of 4
    real, verified test cases). The masked-reconstruction objective alone
    has no incentive to ignore network-specific signal (which router,
    which cloud endpoints, background chatter, etc.) that happens to
    correlate with device identity on the *training* network but is not
    intrinsic to the device.

    This adds a second, adversarial objective, adapted from a technique
    already proven for an analogous problem in RF (radio-signal)
    fingerprinting -- recognizing the same physical device across
    different receiver hardware, a direct sibling of "same device,
    different network" here: alongside the shared encoder, a shared
    `DomainDiscriminator` is also FedAvg'd every round. Each client's
    federated client ID is used as a free domain label (already known
    locally, zero extra labeling or communication cost) to train that
    discriminator to guess which client an embedding came from, while a
    gradient-reversal layer pushes the *encoder* to make embeddings that
    make this progressively harder -- i.e. to stop encoding
    network-identity information, leaving only (hopefully) device-general
    signal.

    Real caveat this training-loop design does NOT resolve: each client
    only ever sees its own domain label locally (no raw data crosses
    federation boundaries), so the discriminator's ability to actually
    distinguish domains depends on FedAvg accumulating signal about
    *other* domains' embedding distributions across rounds, not from any
    single client seeing multiple domains directly. This is the same
    approximation federated domain-adversarial methods in prior work make
    (e.g. FADA); it is not validated as equivalent to a centralized
    domain-adversarial setup here, only run and reported as-is.

    `domain_labels`: optional {client_key: domain_id} override. Default
    (None) is one domain per dict key in `client_sequences` -- the
    original v1 behavior, e.g. one domain per UNSW federated client. A
    v2 real experiment (Section VII-G) used the default with UNSW clients
    plus real YourThings devices added as extra per-device domains, which
    made cross-network results worse (0/4 vs. 1/4), with inconsistent
    domain granularity (UNSW clients mix multiple device types; added
    YourThings domains are single-device) as the leading unconfirmed
    hypothesis for why. Pass `domain_labels` explicitly to collapse
    multiple dict keys into fewer, more consistent domains -- e.g. all
    UNSW clients to domain 0 and all YourThings devices to domain 1, for
    a clean two-network ("which physical network is this from") signal
    instead of a per-client or per-device one.
    """
    torch.manual_seed(seed)
    client_ids = list(client_sequences.keys())
    if domain_labels is not None:
        domain_idx = domain_labels
        n_domains = len(set(domain_labels.values()))
    else:
        n_domains = len(client_ids)
        domain_idx = {cid: i for i, cid in enumerate(client_ids)}

    global_encoder = SelfSupervisedEncoder(seq_len=seq_len, d_model=d_model, embed_dim=embed_dim)
    global_disc = DomainDiscriminator(embed_dim=embed_dim, n_domains=n_domains)

    client_tensors = {
        cid: torch.tensor(X, dtype=torch.float32) for cid, X in client_sequences.items()
    }

    for r in range(num_rounds):
        encoder_states, disc_states = [], []
        for cid, X in client_tensors.items():
            model = SelfSupervisedEncoder(seq_len=seq_len, d_model=d_model, embed_dim=embed_dim)
            fed.set_state(model, fed.get_state(global_encoder))
            disc = DomainDiscriminator(embed_dim=embed_dim, n_domains=n_domains)
            fed.set_state(disc, fed.get_state(global_disc))
            model.train()
            disc.train()
            opt = torch.optim.Adam(list(model.parameters()) + list(disc.parameters()), lr=lr)
            domain_label = torch.full((batch_size,), domain_idx[cid], dtype=torch.long)

            for _ in range(local_epochs):
                idx = torch.randperm(len(X))
                for i in range(0, len(X), batch_size):
                    b = idx[i : i + batch_size]
                    opt.zero_grad()
                    recon_loss = model.forward_pretrain(X[b])

                    emb = model.embed_trainable(X[b])
                    emb_rev = gradient_reversal(emb, grl_lambda)
                    domain_logits = disc(emb_rev)
                    lbl = domain_label[: emb.shape[0]]
                    domain_loss = nn.functional.cross_entropy(domain_logits, lbl)

                    loss = recon_loss + domain_loss_weight * domain_loss
                    loss.backward()
                    opt.step()
            encoder_states.append(fed.get_state(model))
            disc_states.append(fed.get_state(disc))

        weights = [1.0] * len(encoder_states)
        fed.set_state(global_encoder, fed.weighted_average_states(encoder_states, weights))
        fed.set_state(global_disc, fed.weighted_average_states(disc_states, weights))
        print(f"  Phase 0 domain-adversarial pretraining round {r+1}/{num_rounds} done")

    return global_encoder


def contrastive_finetune_encoder(
    encoder: SelfSupervisedEncoder,
    positive_pairs: "Dict[str, tuple]",
    negative_pool: Dict[str, np.ndarray],
    num_steps: int = 300,
    lr: float = 1e-4,
    tau: float = 0.1,
    batch_size: int = 64,
    n_negatives: int = 5,
    seed: int = 0,
) -> SelfSupervisedEncoder:
    """Real, explicit cross-network alignment (Section VII-G's most
    promising untried direction, after domain-adversarial training in
    three configurations all failed to improve on the plain encoder's
    1/4 real cross-network matches): fine-tunes an ALREADY-PRETRAINED
    encoder with a supervised contrastive (InfoNCE-style) loss that
    directly pulls a known cross-network same-device-type pair's
    embeddings together and pushes them away from other device types'
    embeddings.

    This is a genuinely different regime from every other mechanism in
    this paper: it requires real cross-network device-identity labels
    (which UNSW class corresponds to which YourThings device), not
    self-supervision, and it is centralized fine-tuning on a copy of the
    encoder, not federated (comparing two specific clients' embeddings
    directly is incompatible with the federated privacy model this paper
    otherwise maintains throughout -- flagged here rather than silently
    treated as still-federated).

    `positive_pairs`: {class_name: (unsw_X_samples, yourthings_X_samples)}
    for classes we have explicit cross-network alignment supervision for.
    `negative_pool`: {class_name: unsw_X_samples} for ALL UNSW classes
    (including the positive ones, used as negatives for each other).

    HONESTY-CRITICAL: to test whether this generalizes rather than just
    memorizes the exact pairs it was shown, `positive_pairs` must be
    built with the evaluation device explicitly excluded (leave-one-out)
    -- see run_yourthings_onboarding_contrastive_alignment.py, which
    does this for real, 4-fold, rather than aligning on all 4 known pairs
    and testing on the same 4 (which would not be a real test)."""
    torch.manual_seed(seed)
    model = SelfSupervisedEncoder(
        seq_len=encoder.seq_len, embed_dim=encoder.embed_proj.out_features,
    )
    fed.set_state(model, fed.get_state(encoder))
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    rng = np.random.RandomState(seed)
    pos_classes = list(positive_pairs.keys())
    all_neg_classes = list(negative_pool.keys())

    for step in range(num_steps):
        cls = pos_classes[step % len(pos_classes)]
        unsw_X, yt_X = positive_pairs[cls]
        a_idx = rng.choice(len(unsw_X), size=min(batch_size, len(unsw_X)), replace=len(unsw_X) < batch_size)
        p_idx = rng.choice(len(yt_X), size=min(batch_size, len(yt_X)), replace=len(yt_X) < batch_size)
        anchor_batch = torch.tensor(unsw_X[a_idx], dtype=torch.float32)
        positive_batch = torch.tensor(yt_X[p_idx], dtype=torch.float32)

        neg_classes = [c for c in all_neg_classes if c != cls]
        neg_classes = list(rng.choice(neg_classes, size=min(n_negatives, len(neg_classes)), replace=False))

        opt.zero_grad()
        anchor_emb = model.embed_trainable(anchor_batch).mean(dim=0, keepdim=True)
        positive_emb = model.embed_trainable(positive_batch).mean(dim=0, keepdim=True)
        neg_embs = []
        for nc in neg_classes:
            nX = negative_pool[nc]
            n_idx = rng.choice(len(nX), size=min(batch_size, len(nX)), replace=len(nX) < batch_size)
            neg_batch = torch.tensor(nX[n_idx], dtype=torch.float32)
            neg_embs.append(model.embed_trainable(neg_batch).mean(dim=0, keepdim=True))
        neg_emb = torch.cat(neg_embs, dim=0)

        anchor_n = nn.functional.normalize(anchor_emb, dim=-1)
        pos_n = nn.functional.normalize(positive_emb, dim=-1)
        neg_n = nn.functional.normalize(neg_emb, dim=-1)

        pos_sim = (anchor_n * pos_n).sum(-1) / tau
        neg_sim = (anchor_n @ neg_n.t()) / tau
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sim], dim=-1)
        labels = torch.zeros(1, dtype=torch.long)
        loss = nn.functional.cross_entropy(logits, labels)
        loss.backward()
        opt.step()

    return model


def embed_all(encoder: SelfSupervisedEncoder, client_sequences: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    encoder.eval()
    out = {}
    for cid, X in client_sequences.items():
        Xt = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            emb = encoder.embed(Xt)
        out[cid] = emb.numpy()
    return out
