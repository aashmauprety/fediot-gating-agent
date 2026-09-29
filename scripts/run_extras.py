"""Additional real experiments closing several paper TODOs on real UNSW
data: (a) window-length N sweep for the PCA baseline (Section V-B /
VII-B), (b) client-count scalability curve (Section VII-H), (c) per-class
F1 breakdown for both PCA and Phase 0 encoder at the default N=100
config (Section VII-B). All real, all on real UNSW device-identification
data, same protocol style as run_unsw.py.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

from fedgate.data import load_unsw_device_identification
from fedgate.classifier_training import federated_train_classifier

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def pca_baseline_f1(clients, classes, num_rounds=15, seed=0):
    client_seqs = {cid: c["X"] for cid, c in clients.items()}
    client_labels = {cid: c["y_device"] for cid, c in clients.items()}
    all_X = np.concatenate(list(client_seqs.values()), axis=0)
    n_components = min(80, all_X.shape[1])
    pca = PCA(n_components=n_components, random_state=seed)
    pca.fit(all_X)
    client_pca_emb = {cid: pca.transform(X).astype(np.float32) for cid, X in client_seqs.items()}
    result = federated_train_classifier(
        client_pca_emb, client_labels, num_classes=len(classes), embed_dim=n_components,
        num_rounds=num_rounds, seed=seed,
    )
    return result


def window_length_sweep():
    print("=== (a) Window-length N sweep, PCA baseline (real UNSW) ===")
    results = []
    for N in (50, 100, 200, 500):
        t0 = time.time()
        clients = load_unsw_device_identification(
            seq_len=N, max_windows_per_device=2000, min_windows_per_device=100,
            n_clients=10, seed=0,
        )
        classes = clients["unsw_client_0"]["classes"]
        load_time = time.time() - t0
        t0 = time.time()
        result = pca_baseline_f1(clients, classes, num_rounds=15, seed=0)
        train_time = time.time() - t0
        total_windows = sum(len(c["y_device"]) for c in clients.values())
        row = {
            "N": N,
            "num_classes": len(classes),
            "total_windows": total_windows,
            "final_macro_f1": result["final_macro_f1"],
            "load_time_s": load_time,
            "train_time_s": train_time,
        }
        results.append(row)
        print(f"  N={N}: classes={len(classes)} windows={total_windows} "
              f"F1={result['final_macro_f1']:.4f} (load={load_time:.1f}s train={train_time:.1f}s)")
    with open(os.path.join(RESULTS_DIR, "unsw_n_sweep.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def client_scalability():
    print("\n=== (b) Client-count scalability curve, PCA baseline (real UNSW) ===")
    results = []
    for n_clients in (2, 5, 10, 20, 50):
        t0 = time.time()
        clients = load_unsw_device_identification(
            seq_len=100, max_windows_per_device=2000, min_windows_per_device=100,
            n_clients=n_clients, seed=0,
        )
        classes = clients["unsw_client_0"]["classes"]
        result = pca_baseline_f1(clients, classes, num_rounds=15, seed=0)
        elapsed = time.time() - t0
        row = {
            "n_clients": n_clients,
            "final_macro_f1": result["final_macro_f1"],
            "wall_clock_s": elapsed,
        }
        results.append(row)
        print(f"  n_clients={n_clients}: F1={result['final_macro_f1']:.4f} (elapsed {elapsed:.1f}s)")
    with open(os.path.join(RESULTS_DIR, "unsw_client_scalability.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def per_device_breakdown():
    print("\n=== (c) Per-device F1 breakdown, PCA vs Phase 0 encoder (real UNSW) ===")
    from fedgate.phase0_training import federated_pretrain_encoder, embed_all

    clients = load_unsw_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=100,
        n_clients=10, seed=0,
    )
    classes = clients["unsw_client_0"]["classes"]
    client_labels = {cid: c["y_device"] for cid, c in clients.items()}

    pca_result = pca_baseline_f1(clients, classes, num_rounds=25, seed=0)
    pca_report = classification_report(
        pca_result["final_true"], pca_result["final_preds"], target_names=classes,
        output_dict=True, zero_division=0,
    )

    client_seqs = {cid: c["X"] for cid, c in clients.items()}
    print("  training Phase 0 encoder (15 rounds) for per-device breakdown ...")
    encoder = federated_pretrain_encoder(client_seqs, seq_len=100, num_rounds=15, embed_dim=32, seed=0)
    client_emb = embed_all(encoder, client_seqs)
    enc_result = federated_train_classifier(
        client_emb, client_labels, num_classes=len(classes), embed_dim=32, num_rounds=25, seed=0,
    )
    enc_report = classification_report(
        enc_result["final_true"], enc_result["final_preds"], target_names=classes,
        output_dict=True, zero_division=0,
    )

    rows = []
    for cls in classes:
        rows.append({
            "device": cls,
            "pca_f1": pca_report[cls]["f1-score"],
            "pca_support": pca_report[cls]["support"],
            "encoder_f1": enc_report[cls]["f1-score"],
            "encoder_support": enc_report[cls]["support"],
        })
        print(f"  {cls}: PCA F1={pca_report[cls]['f1-score']:.3f}  Encoder F1={enc_report[cls]['f1-score']:.3f}")

    out = {
        "rows": rows,
        "pca_macro_f1": pca_result["final_macro_f1"],
        "encoder_macro_f1": enc_result["final_macro_f1"],
    }
    with open(os.path.join(RESULTS_DIR, "unsw_per_device_f1.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    window_length_sweep()
    client_scalability()
    per_device_breakdown()
    print("\nAll extras complete.")
