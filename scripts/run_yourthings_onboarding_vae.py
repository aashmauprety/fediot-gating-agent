"""Real test of VAE-style latent regularization for the Phase 0 encoder
(Section VII-G's cross-network generalization investigation), replicating
a controlled ablation from Sivanathan et al., "Generalizable IoT Traffic
Representations for Cross-Network Device Identification" (arXiv:2601.19315,
already cited in Related Work): same architecture and features, only
difference is a KL-divergence term added to the training loss (turning
the deterministic autoencoder into a VAE). Their controlled ablation
found this was the dominant factor in cross-environment robustness in
their setting -- more than feature richness or model scale -- with a
real, quantified 6-9 point downstream macro-F1 gap under distribution
shift.

Identical evaluation protocol to run_yourthings_onboarding_v2.py (the
plain-encoder baseline this compares against): same UNSW data, same 15
pretraining rounds, same KB-building and query procedure, same 4 real
device-identity alignment pairs. Only the encoder's training objective
differs (variational=True here).
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import numpy as np

from fedgate.data import load_unsw_device_identification, load_yourthings_device_identification
from fedgate.phase0_training import federated_pretrain_encoder, embed_all
from fedgate.onboarding import DeviceFingerprintKB, RuleBasedOnboardingPolicy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

YOURTHINGS_TO_UNSW_ALIGNMENT = {
    "SamsungSmartThingsHub": "Smart Things",
    "AmazonEchoGen1": "Amazon Echo",
    "BelkinWeMoMotionSensor": "Belkin wemo motion sensor",
    "LIFXVirtualBulb": "Light Bulbs LiFX Smart Bulb",
}


def main(phase0_rounds=15, kl_weight=0.001):
    print("Loading real UNSW data ...")
    unsw_clients = load_unsw_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=100,
        n_clients=10, seed=0,
    )
    unsw_classes = unsw_clients["unsw_client_0"]["classes"]
    unsw_seqs = {cid: c["X"] for cid, c in unsw_clients.items()}
    unsw_labels = {cid: c["y_device"] for cid, c in unsw_clients.items()}

    print(f"\nTraining VAE-style Phase 0 encoder federatedly on UNSW "
          f"({phase0_rounds} rounds, kl_weight={kl_weight}) ...")
    t0 = time.time()
    encoder = federated_pretrain_encoder(
        unsw_seqs, seq_len=100, num_rounds=phase0_rounds, embed_dim=32,
        seed=0, variational=True, kl_weight=kl_weight,
    )
    print(f"Done in {time.time()-t0:.1f}s")

    print("\nBuilding device-fingerprint KB from UNSW embeddings (VAE mean, deterministic) ...")
    unsw_emb = embed_all(encoder, unsw_seqs)
    per_class_embeddings = {c: [] for c in unsw_classes}
    for cid in unsw_clients:
        emb = unsw_emb[cid]
        labels = unsw_labels[cid]
        for i, lbl in enumerate(labels):
            per_class_embeddings[unsw_classes[lbl]].append(emb[i])
    kb = DeviceFingerprintKB()
    for cls, embs in per_class_embeddings.items():
        if not embs:
            continue
        kb.add(label=cls, embedding=np.mean(np.stack(embs), axis=0))
    print(f"KB has {len(kb.entries)} admitted device fingerprints")

    print("\nLoading real YourThings data (IP-verified devices) ...")
    yt_clients = load_yourthings_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=5,
        n_clients=1, seed=0,
    )
    yt_classes = yt_clients["yourthings_client_0"]["classes"]
    yt_X_all = yt_clients["yourthings_client_0"]["X"]
    yt_y_all = yt_clients["yourthings_client_0"]["y_device"]

    print("\nEmbedding YourThings windows through the VAE-trained encoder ...")
    yt_emb = embed_all(encoder, {"yt": yt_X_all})["yt"]

    policy = RuleBasedOnboardingPolicy(k=5)
    results = []
    for cls_idx, cls_name in enumerate(yt_classes):
        mask = yt_y_all == cls_idx
        if mask.sum() == 0:
            continue
        query_embedding = yt_emb[mask].mean(axis=0)
        action, rationale, confidence, predicted_label = policy.decide(kb, query_embedding)
        aligned_unsw_class = YOURTHINGS_TO_UNSW_ALIGNMENT.get(cls_name)
        row = {
            "yourthings_device": cls_name,
            "n_windows": int(mask.sum()),
            "aligned_unsw_class": aligned_unsw_class,
            "is_real_cross_network_same_device_test": aligned_unsw_class is not None,
            "action": action,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct_prediction_if_known": (
                (predicted_label == aligned_unsw_class) if aligned_unsw_class else None
            ),
        }
        results.append(row)
        if aligned_unsw_class:
            print(f"\n{cls_name} <-> UNSW '{aligned_unsw_class}':")
            print(f"  action={action}, predicted={predicted_label}, confidence={confidence:.2%}, "
                  f"correct={row['correct_prediction_if_known']}")

    n_aligned = sum(1 for r in results if r["is_real_cross_network_same_device_test"])
    n_correct = sum(1 for r in results if r["correct_prediction_if_known"])
    n_auto_admit_unaligned = sum(
        1 for r in results if not r["is_real_cross_network_same_device_test"] and r["action"] == "auto-admit"
    )
    print(f"\n=== Summary: {n_correct}/{n_aligned} real cross-network matches correct "
          f"(plain encoder baseline: 1/4; all 3 domain-adv variants: 0-1/4; "
          f"contrastive alignment: 1/4). False auto-admits: {n_auto_admit_unaligned} ===")

    out = {
        "description": "VAE-style latent-regularized encoder: cross-dataset test, UNSW-trained "
                        "Phase0 encoder (masked-recon + KL regularization), real IP-verified YourThings traffic",
        "kl_weight": kl_weight,
        "unsw_classes": unsw_classes,
        "yourthings_identified_devices": yt_classes,
        "alignment_used": YOURTHINGS_TO_UNSW_ALIGNMENT,
        "results": results,
        "n_real_cross_network_tests": n_aligned,
        "n_correct": n_correct,
        "n_false_auto_admit_unaligned": n_auto_admit_unaligned,
    }
    path = os.path.join(RESULTS_DIR, "yourthings_cross_network_vae.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
