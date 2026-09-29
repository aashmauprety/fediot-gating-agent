"""Real experiments on real UNSW data (Sections VII-A, VII-B):
  (A) Phase I device-type head trained on a PCA(100->80) baseline
      (matching the paper's original/interim feature pipeline).
  (B) Phase 0 federated self-supervised encoder pretraining, then the
      same Phase I head trained on the encoder's embeddings instead.
Both use the SAME 10-client partition, same train/test splits, same
classifier architecture, same number of rounds -- an apples-to-apples
comparison of representation quality, which is the paper's headline
representation-learning claim (Section VII-B).

No Gating Agent, no attacks, in this script -- purely the backbone /
representation comparison.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import numpy as np
from sklearn.decomposition import PCA

from fedgate.data import load_unsw_device_identification
from fedgate.classifier_training import federated_train_classifier
from fedgate.phase0_training import federated_pretrain_encoder, embed_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main(num_rounds=25, phase0_rounds=15, min_windows=100, n_clients=10):
    print("Loading real UNSW device-identification data ...")
    t0 = time.time()
    clients = load_unsw_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=min_windows,
        n_clients=n_clients, seed=0,
    )
    classes = clients["unsw_client_0"]["classes"]
    print(f"Loaded {len(clients)} clients, {len(classes)} device classes, in {time.time()-t0:.1f}s")
    print(f"Classes: {classes}")

    client_seqs = {cid: c["X"] for cid, c in clients.items()}
    client_labels = {cid: c["y_device"] for cid, c in clients.items()}
    total_windows = sum(len(y) for y in client_labels.values())
    print(f"Total windows: {total_windows}")

    # === (A) PCA(100->80) baseline, matching the paper's interim pipeline ===
    print("\n=== (A) PCA baseline backbone (Section VII-A) ===")
    all_X = np.concatenate(list(client_seqs.values()), axis=0)
    pca = PCA(n_components=80, random_state=0)
    pca.fit(all_X)
    client_pca_emb = {cid: pca.transform(X).astype(np.float32) for cid, X in client_seqs.items()}

    t0 = time.time()
    pca_result = federated_train_classifier(
        client_pca_emb, client_labels, num_classes=len(classes), embed_dim=80,
        num_rounds=num_rounds, seed=0,
    )
    print(f"PCA baseline: final macro-F1 = {pca_result['final_macro_f1']:.4f} "
          f"({time.time()-t0:.1f}s)")

    # === (B) Phase 0 federated self-supervised encoder ===
    print("\n=== (B) Phase 0 federated self-supervised encoder (Section VII-B) ===")
    t0 = time.time()
    encoder = federated_pretrain_encoder(
        client_seqs, seq_len=100, num_rounds=phase0_rounds, embed_dim=32, seed=0,
    )
    print(f"Phase 0 pretraining done in {time.time()-t0:.1f}s")

    client_encoder_emb = embed_all(encoder, client_seqs)
    t0 = time.time()
    encoder_result = federated_train_classifier(
        client_encoder_emb, client_labels, num_classes=len(classes), embed_dim=32,
        num_rounds=num_rounds, seed=0,
    )
    print(f"Phase 0 encoder: final macro-F1 = {encoder_result['final_macro_f1']:.4f} "
          f"({time.time()-t0:.1f}s)")

    out = {
        "description": "Real UNSW device identification, 10 federated clients, 18 device classes",
        "num_classes": len(classes),
        "classes": classes,
        "total_windows": total_windows,
        "pca_baseline": {
            "embed_dim": 80,
            "final_macro_f1": pca_result["final_macro_f1"],
            "rounds": pca_result["rounds"],
        },
        "phase0_encoder": {
            "embed_dim": 32,
            "phase0_rounds": phase0_rounds,
            "final_macro_f1": encoder_result["final_macro_f1"],
            "rounds": encoder_result["rounds"],
        },
    }
    path = os.path.join(RESULTS_DIR, "unsw_backbone_comparison.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")
    print(f"\nSUMMARY: PCA baseline F1={pca_result['final_macro_f1']:.4f} vs. "
          f"Phase 0 encoder F1={encoder_result['final_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
