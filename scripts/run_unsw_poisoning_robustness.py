"""Real experiment: does plain FL degrade under poisoning for the
DEVICE-IDENTIFICATION task, not just the intrusion-detection task?

This has never been tested in this codebase. Section VII-A/B validated
the federated self-supervised encoder's representation quality with NO
attacker present. This script applies the SAME threat model already
used for the attack-type/IDS head (Section IV: untargeted label-flip,
targeted label-flip, both combined with the same magnitude-scaling
model-poisoning boost from `scale_update`) to the device-type head, on
real UNSW device-identification traffic, plain FedAvg, no defense.

No literature we found (including He et al. 2021, "Edge Device
Identification Based on Federated Learning and Network Traffic Feature
Engineering", IEEE TCCN, and Sanchez Sanchez et al. 2021, arXiv
2111.14434, the two closest FL-device-ID papers) tests this specific
combination for network-traffic-based device-type classification.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from fedgate.data import load_unsw_device_identification
from fedgate.phase0_training import federated_pretrain_encoder, embed_all
from fedgate.models import ClassifierHead
from fedgate import federated as fed
from fedgate.attacks import label_flip_untargeted, label_flip_targeted, scale_update

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_poisoned_fedavg(
    client_embeddings, client_labels, num_classes, embed_dim,
    malicious_ids, attack_type, source_class=None, target_class=None,
    num_rounds=25, lr=1e-3, local_epochs=1, batch_size=64,
    train_frac=0.8, val_frac=0.1, seed=0, scale_factor=5.0,
):
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

    final_f1 = None
    for r in range(num_rounds):
        client_states = []
        prev = fed.get_state(global_model)
        for cid, s in splits.items():
            Xtr, ytr = s["train"]
            Xt = torch.tensor(Xtr, dtype=torch.float32)
            yt = torch.tensor(ytr, dtype=torch.long)

            if cid in malicious_ids:
                if attack_type == "untargeted":
                    yt = label_flip_untargeted(yt, num_classes, rng)
                elif attack_type == "targeted":
                    yt = label_flip_targeted(yt, source_class, target_class)

            def data_iter():
                yield from fed.batches(Xt, yt, batch_size)

            local_model = fed.local_train_step(
                global_model, loss_fn, torch.optim.Adam, lr, local_epochs, data_iter
            )
            state = fed.get_state(local_model)
            if cid in malicious_ids:
                delta = fed.state_delta(state, prev)
                delta = scale_update(delta, scale_factor)
                state = {k: prev[k] + delta[k].to(prev[k].dtype) for k in prev}
            client_states.append(state)
        new_global = fed.weighted_average_states(client_states, [1.0] * len(client_states))
        fed.set_state(global_model, new_global)

        global_model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for cid, s in splits.items():
                Xte, yte = s["test"]
                Xt = torch.tensor(Xte, dtype=torch.float32)
                logits = global_model(Xt)
                preds = logits.argmax(dim=1).numpy()
                all_preds.append(preds)
                all_true.append(yte)
        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        final_f1 = f1_score(all_true, all_preds, average="macro")

    success_rate = None
    if attack_type == "targeted":
        mask = all_true == source_class
        if mask.sum() > 0:
            success_rate = float((all_preds[mask] == target_class).mean())

    return {"final_macro_f1": float(final_f1), "targeted_success_rate": success_rate}


def main(num_rounds=25, min_windows=100, n_clients=10, phase0_rounds=15, seed=0):
    print("Loading real UNSW device-identification data ...")
    t0 = time.time()
    clients = load_unsw_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=min_windows,
        n_clients=n_clients, seed=seed,
    )
    classes = clients["unsw_client_0"]["classes"]
    num_classes = len(classes)
    print(f"Loaded {len(clients)} clients, {num_classes} device classes in {time.time()-t0:.1f}s")

    client_seqs = {cid: c["X"] for cid, c in clients.items()}
    client_labels = {cid: c["y_device"] for cid, c in clients.items()}
    client_ids = list(clients.keys())

    print("\nPretraining federated self-supervised encoder (Phase 0) ...")
    t0 = time.time()
    encoder = federated_pretrain_encoder(
        client_seqs, seq_len=100, num_rounds=phase0_rounds, embed_dim=32, seed=seed,
    )
    client_emb = embed_all(encoder, client_seqs)
    print(f"Phase 0 done in {time.time()-t0:.1f}s")

    source_class, target_class = 0, 1
    print(f"Targeted attack: device class {classes[source_class]!r} -> {classes[target_class]!r}")

    results = {}
    for attack_type in ("untargeted", "targeted"):
        for alpha in (0, 1, 2, 3, 4):
            malicious_ids = set(client_ids[:alpha])
            t0 = time.time()
            r = run_poisoned_fedavg(
                client_emb, client_labels, num_classes, embed_dim=32,
                malicious_ids=malicious_ids, attack_type=attack_type,
                source_class=source_class, target_class=target_class,
                num_rounds=num_rounds, seed=seed,
            )
            dt = time.time() - t0
            key = f"{attack_type}_alpha{alpha}"
            results[key] = r
            print(f"[{key}] macro-F1={r['final_macro_f1']:.4f} "
                  f"success_rate={r['targeted_success_rate']} ({dt:.1f}s)")

    out_path = os.path.join(RESULTS_DIR, "unsw_deviceid_poisoning_robustness.json")
    with open(out_path, "w") as f:
        json.dump({"classes": classes, "n_clients": n_clients, "results": results}, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
