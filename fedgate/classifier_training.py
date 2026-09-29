"""Minimal federated classifier training loop (no attack, no gating) --
used to train the device-type head on real UNSW data, for both the PCA
baseline and the Phase 0 encoder embeddings (Sections VII-A, VII-B).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from . import federated as fed
from .models import ClassifierHead


def federated_train_classifier(
    client_embeddings: Dict[str, np.ndarray],
    client_labels: Dict[str, np.ndarray],
    num_classes: int,
    embed_dim: int,
    num_rounds: int = 25,
    lr: float = 1e-3,
    local_epochs: int = 1,
    batch_size: int = 64,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    seed: int = 0,
    return_model: bool = False,
) -> Dict:
    """Train a ClassifierHead federatedly over the given clients'
    embeddings, plain FedAvg (no attack, no gating). Returns per-round
    macro-F1 on a held-out test split.

    `return_model=True` additionally returns the trained model, the
    final round's softmax probabilities (not just argmax predictions),
    and the raw test features -- needed for confidence-gated LLM
    escalation (real onboarding/triage use case), which needs to see
    *how* confident the classifier was, not just its final guess."""
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    splits = {}
    for cid, X in client_embeddings.items():
        y = client_labels[cid]
        n = len(y)
        perm = rng.permutation(n)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        splits[cid] = {
            "train": (X[perm[:n_train]], y[perm[:n_train]]),
            "test": (X[perm[n_train + n_val:]], y[perm[n_train + n_val:]]),
        }

    global_model = ClassifierHead(embed_dim, num_classes)
    loss_fn = nn.CrossEntropyLoss()

    logs = []
    for r in range(num_rounds):
        client_states = []
        for cid, s in splits.items():
            Xtr, ytr = s["train"]
            Xt = torch.tensor(Xtr, dtype=torch.float32)
            yt = torch.tensor(ytr, dtype=torch.long)

            def data_iter():
                yield from fed.batches(Xt, yt, batch_size)

            local_model = fed.local_train_step(
                global_model, loss_fn, torch.optim.Adam, lr, local_epochs, data_iter
            )
            client_states.append(fed.get_state(local_model))
        new_global = fed.weighted_average_states(client_states, [1.0] * len(client_states))
        fed.set_state(global_model, new_global)

        global_model.eval()
        all_preds, all_true, all_probs, all_X = [], [], [], []
        with torch.no_grad():
            for cid, s in splits.items():
                Xte, yte = s["test"]
                logits = global_model(torch.tensor(Xte, dtype=torch.float32))
                probs = torch.softmax(logits, dim=1)
                all_preds.append(logits.argmax(dim=1).numpy())
                all_probs.append(probs.numpy())
                all_X.append(Xte)
                all_true.append(yte)
        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        f1 = f1_score(all_true, all_preds, average="macro", zero_division=0)
        logs.append({"round": r, "macro_f1": float(f1)})

    out = {"rounds": logs, "final_macro_f1": logs[-1]["macro_f1"], "final_preds": all_preds.tolist(), "final_true": all_true.tolist()}
    if return_model:
        out["model"] = global_model
        out["final_probs"] = np.concatenate(all_probs, axis=0)
        out["final_test_X"] = np.concatenate(all_X, axis=0)
    return out
