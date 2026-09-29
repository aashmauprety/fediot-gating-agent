"""Real cross-network generalization + zero-shot onboarding test
(Sections VII-B "cross-network generalization" and VII-G), now finally
possible with two real datasets:

  - The Phase 0 encoder is (re-)trained federatedly on real UNSW data
    (same protocol as run_unsw.py; retrained here rather than loaded from
    disk since run_unsw.py didn't persist weights -- deterministic given
    the same seed, so this reproduces the same encoder).
  - A device-fingerprint knowledge base is built from UNSW's 18 admitted
    device types' embeddings (the "already onboarded" devices).
  - Real YourThings traffic for 3 vendor-identified devices (Samsung
    Smart Things Hub, Philips Hue Hub, Bose SoundTouch 10 -- see
    fedgate/data.py docstring for the vendor-matching caveat) is embedded
    through the SAME UNSW-trained encoder and queried against the KB.

This sets up a real, non-trivial test: UNSW's own device list already
contains "Smart Things" (Samsung SmartThings Hub) as one of its 18
classes, so the YourThings "Samsung Smart Things Hub" traffic is a
same-device-type-different-network query (a real test of whether the
encoder's embedding generalizes across networks) -- while "Philips Hue
Hub" and "Bose SoundTouch 10" are genuinely new device types not in
UNSW's label space at all, the correct onboarding behavior for both is
NOT to auto-admit under an existing label.
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


def main(phase0_rounds=15):
    print("Loading real UNSW data ...")
    unsw_clients = load_unsw_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=100,
        n_clients=10, seed=0,
    )
    unsw_classes = unsw_clients["unsw_client_0"]["classes"]
    print(f"UNSW: {len(unsw_classes)} device classes: {unsw_classes}")

    unsw_seqs = {cid: c["X"] for cid, c in unsw_clients.items()}
    unsw_labels = {cid: c["y_device"] for cid, c in unsw_clients.items()}

    print(f"\nTraining Phase 0 encoder federatedly on UNSW ({phase0_rounds} rounds) ...")
    t0 = time.time()
    encoder = federated_pretrain_encoder(unsw_seqs, seq_len=100, num_rounds=phase0_rounds, embed_dim=32, seed=0)
    print(f"Done in {time.time()-t0:.1f}s")

    # Build device-fingerprint KB: one centroid embedding per UNSW device class
    print("\nBuilding device-fingerprint KB from UNSW embeddings ...")
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
        centroid = np.mean(np.stack(embs), axis=0)
        kb.add(label=cls, embedding=centroid)
    print(f"KB has {len(kb.entries)} admitted device fingerprints: {[e.label for e in kb.entries]}")

    print("\nLoading real YourThings data (vendor-identified devices) ...")
    yt_clients = load_yourthings_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=5,
        n_clients=1, seed=0,
    )
    yt_classes = yt_clients["yourthings_client_0"]["classes"]
    print(f"YourThings: {len(yt_classes)} identified devices: {yt_classes}")

    yt_X = yt_clients["yourthings_client_0"]["X"]
    yt_y = yt_clients["yourthings_client_0"]["y_device"]

    print("\nEmbedding YourThings windows through the UNSW-trained encoder ...")
    yt_emb = embed_all(encoder, {"yt": yt_X})["yt"]

    policy = RuleBasedOnboardingPolicy(k=5)
    results = []
    for cls_idx, cls_name in enumerate(yt_classes):
        mask = yt_y == cls_idx
        if mask.sum() == 0:
            continue
        query_embedding = yt_emb[mask].mean(axis=0)
        action, rationale, confidence, predicted_label = policy.decide(kb, query_embedding)
        in_unsw_label_space = cls_name in unsw_classes
        row = {
            "yourthings_device": cls_name,
            "n_windows": int(mask.sum()),
            "in_unsw_label_space": in_unsw_label_space,
            "action": action,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "rationale": rationale,
            "correct_prediction_if_known": (predicted_label == cls_name) if in_unsw_label_space else None,
        }
        results.append(row)
        print(f"\n{cls_name} (n={mask.sum()}, in UNSW label space: {in_unsw_label_space}):")
        print(f"  action={action}, predicted={predicted_label}, confidence={confidence:.2%}")
        print(f"  rationale: {rationale}")

    out = {
        "description": "Real cross-dataset test: UNSW-trained Phase 0 encoder + device-fingerprint KB, queried with real YourThings traffic",
        "unsw_classes": unsw_classes,
        "yourthings_identified_devices": yt_classes,
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, "yourthings_cross_network_onboarding.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
