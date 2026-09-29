"""Model architectures for FedIoT-Gate: Phase 0 self-supervised encoder and
Phase I classifier heads (Section V-C, VI-A of the paper).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class GradientReversalFunction(torch.autograd.Function):
    """Identity on the forward pass; negates (and scales by `lambda_`) the
    gradient on the backward pass. Standard domain-adversarial training
    (DANN) trick: lets a domain discriminator be trained normally (to get
    better at telling domains apart) while the encoder feeding it gets
    pushed the opposite direction (to make its features *harder* to tell
    apart), all with one shared loss and one optimizer step -- no need to
    alternate min/max steps manually."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def gradient_reversal(x: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, lambda_)


class DomainDiscriminator(nn.Module):
    """Small MLP that tries to predict which federated client (domain) an
    embedding came from. Used only during Phase 0 pretraining, behind a
    gradient-reversal layer -- never used at inference, and not part of
    the frozen encoder Phase I consumes. See
    `federated_pretrain_encoder_domain_adversarial` in phase0_training.py
    for how it's trained federatedly."""

    def __init__(self, embed_dim: int, n_domains: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: pe[:, 1::2].shape[1]])
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class SelfSupervisedEncoder(nn.Module):
    """Phase 0 federated self-supervised traffic encoder (Section V-C).

    Masked-packet reconstruction objective: given a packet-length/direction
    sequence with a random span masked out, reconstruct the masked span
    from context. The encoder itself (everything up to and including the
    mean-pooled embedding) is what gets frozen and reused by Phase I.

    `variational=True` turns this into a VAE-style encoder (Section VII-G's
    cross-network generalization investigation): instead of one
    deterministic embed_proj, the pooled representation is split into a
    mean and log-variance head, a latent vector is sampled via the
    reparameterization trick, and a KL-divergence term (toward a unit
    Gaussian) is added to the training loss alongside reconstruction.
    This is a direct, real replication of a controlled ablation in
    Sivanathan et al., "Generalizable IoT Traffic Representations for
    Cross-Network Device Identification" (arXiv:2601.19315) -- already
    cited in Related Work -- which found that adding exactly this KL
    regularization term (same architecture and features otherwise
    unchanged) was the dominant factor in cross-environment robustness in
    their controlled experiment, well beyond feature richness or model
    scale: their VAE variant's reconstruction error rose only slightly
    under a real distribution-shift test where a deterministic
    autoencoder's rose ~10x, and downstream macro-F1 under shift improved
    by 6-9 points from that one change. At inference (embed()), the mean
    is used deterministically, as is standard for VAE embeddings; the
    sampling and KL term are training-time only.
    """

    def __init__(
        self,
        seq_len: int = 100,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        embed_dim: int = 32,
        mask_frac: float = 0.15,
        variational: bool = False,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.mask_frac = mask_frac
        self.variational = variational
        self.input_proj = nn.Linear(1, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=seq_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        if variational:
            self.mu_proj = nn.Linear(d_model, embed_dim)
            self.logvar_proj = nn.Linear(d_model, embed_dim)
        else:
            self.embed_proj = nn.Linear(d_model, embed_dim)
        self.recon_head = nn.Linear(d_model, 1)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

    def _embed_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L) raw signed packet-length sequence -> (B, L, d_model)
        h = self.input_proj(x.unsqueeze(-1))
        h = self.pos_enc(h)
        return h

    def _latent(self, pooled: torch.Tensor):
        """Returns (z, mu, logvar) if variational (z sampled via
        reparameterization during training), else (embed, None, None)."""
        if not self.variational:
            return self.embed_proj(pooled), None, None
        mu = self.mu_proj(pooled)
        logvar = self.logvar_proj(pooled)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z, mu, logvar

    def forward_pretrain(self, x: torch.Tensor, kl_weight: float = 0.001):
        """One self-supervised training step: mask a random span, predict
        it. If variational, also adds a KL-divergence term toward a unit
        Gaussian, scaled by `kl_weight` (Sivanathan et al. select 0.001 as
        their "balanced" setting via grid search; not separately tuned
        here). Returns total loss (reconstruction [+ KL if variational])."""
        B, L = x.shape
        device = x.device
        mask_len = max(1, int(L * self.mask_frac))
        starts = torch.randint(0, L - mask_len + 1, (B,), device=device)
        mask = torch.zeros(B, L, dtype=torch.bool, device=device)
        for i in range(B):
            mask[i, starts[i] : starts[i] + mask_len] = True

        tokens = self._embed_tokens(x)
        tokens = torch.where(mask.unsqueeze(-1), self.mask_token, tokens)
        h = self.transformer(tokens)
        pred = self.recon_head(h).squeeze(-1)  # (B, L)
        recon_loss = ((pred - x) ** 2 * mask.float()).sum() / mask.float().sum().clamp(min=1)

        if not self.variational:
            return recon_loss

        pooled = h.mean(dim=1)
        _, mu, logvar = self._latent(pooled)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
        return recon_loss + kl_weight * kl

    @torch.no_grad()
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Frozen inference path: raw sequence -> pooled embedding. Uses
        the mean deterministically when variational (standard for VAE
        embeddings at inference -- no sampling noise at eval time)."""
        tokens = self._embed_tokens(x)
        h = self.transformer(tokens)
        pooled = h.mean(dim=1)
        if self.variational:
            return self.mu_proj(pooled)
        return self.embed_proj(pooled)

    def embed_trainable(self, x: torch.Tensor) -> torch.Tensor:
        """Same as embed() but keeps the graph (used only if someone wants
        to fine-tune the encoder; Phase I as designed does NOT use this --
        it freezes the encoder and calls embed())."""
        tokens = self._embed_tokens(x)
        h = self.transformer(tokens)
        pooled = h.mean(dim=1)
        if self.variational:
            return self.mu_proj(pooled)
        return self.embed_proj(pooled)


class ClassifierHead(nn.Module):
    """Phase I classifier head (Section VI-A): a small Transformer encoder
    over the embedding (treated as a length-1 "sequence", consistent with
    using only the Transformer's encoder side per the paper) followed by
    two fully connected layers, one instance per task (device-type /
    attack-type), independent parameters, same architecture.

    For raw feature-vector inputs (e.g. N-BaIoT's 115-dim flow features,
    which bypass the Phase 0 raw-sequence encoder -- see data.py), the
    "embedding" is just that feature vector itself.
    """

    def __init__(self, embed_dim: int, num_classes: int, fc_hidden: int = 64):
        super().__init__()
        # embed_dim must be divisible by nhead; guard for small/odd dims
        nhead = 1
        for cand in (8, 4, 2, 1):
            if embed_dim % cand == 0:
                nhead = cand
                break
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=embed_dim * 2,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=2)
        self.fc1 = nn.Linear(embed_dim, fc_hidden)
        self.fc2 = nn.Linear(fc_hidden, num_classes)
        self.act = nn.ReLU()

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        # embedding: (B, D) -> treat as a length-1 sequence for the encoder
        h = self.transformer(embedding.unsqueeze(1)).squeeze(1)
        h = self.act(self.fc1(h))
        logits = self.fc2(h)
        return logits
