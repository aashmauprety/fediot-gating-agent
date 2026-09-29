"""Updated cross-network generalization + zero-shot onboarding test
(Sections VII-B, VII-G), using the real, verified YourThings
device_mapping.csv (device name -> local IP) discovered after the
original run_yourthings_onboarding.py was written -- see
fedgate/data.py's module docstring for the full story. That script
identified only 3 devices via a weaker vendor-OUI guess; this one uses
real ground truth and typically identifies 40+ devices (however many are
active with enough packets in the processed day(s)).

New in this version: an explicit, manually-verified alignment between
YourThings' device_mapping.csv names and UNSW's 18 class names, checked
by actual product identity (not string matching, since the two datasets
use different naming conventions for the same products) --
YOURTHINGS_TO_UNSW_ALIGNMENT below. Confirmed real overlaps: Smart
Things (Samsung SmartThings Hub), Amazon Echo, Belkin wemo motion
sensor, and Light Bulbs LiFX Smart Bulb -- four genuine same-product
cross-network tests instead of the original script's one.
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

# Manually verified by actual product identity (checked against UNSW's
# exact class list and YourThings' device_mapping.csv), not string
# matching -- the two datasets name the same products differently.
YOURTHINGS_TO_UNSW_ALIGNMENT = {
    "SamsungSmartThingsHub": "Smart Things",
    "AmazonEchoGen1": "Amazon Echo",
    "BelkinWeMoMotionSensor": "Belkin wemo motion sensor",
    "LIFXVirtualBulb": "Light Bulbs LiFX Smart Bulb",
}


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

    print("\nLoading real YourThings data (IP-verified devices) ...")
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
        aligned_unsw_class = YOURTHINGS_TO_UNSW_ALIGNMENT.get(cls_name)
        row = {
            "yourthings_device": cls_name,
            "n_windows": int(mask.sum()),
            "aligned_unsw_class": aligned_unsw_class,
            "is_real_cross_network_same_device_test": aligned_unsw_class is not None,
            "action": action,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "rationale": rationale,
            "correct_prediction_if_known": (
                (predicted_label == aligned_unsw_class) if aligned_unsw_class else None
            ),
        }
        results.append(row)
        tag = f" <-> UNSW '{aligned_unsw_class}'" if aligned_unsw_class else ""
        print(f"\n{cls_name} (n={mask.sum()}){tag}:")
        print(f"  action={action}, predicted={predicted_label}, confidence={confidence:.2%}")
        print(f"  rationale: {rationale}")

    n_aligned = sum(1 for r in results if r["is_real_cross_network_same_device_test"])
    n_correct = sum(1 for r in results if r["correct_prediction_if_known"])
    print(f"\n=== Summary: {len(results)} devices tested, {n_aligned} are real "
          f"same-product cross-network tests, {n_correct}/{n_aligned} correctly matched ===")

    out = {
        "description": "Real cross-dataset test v2: UNSW-trained Phase 0 encoder + device-fingerprint KB, "
                        "queried with real IP-verified YourThings traffic, manually-aligned product identities",
        "unsw_classes": unsw_classes,
        "yourthings_identified_devices": yt_classes,
        "alignment_used": YOURTHINGS_TO_UNSW_ALIGNMENT,
        "results": results,
        "n_real_cross_network_tests": n_aligned,
        "n_correct": n_correct,
    }
    path = os.path.join(RESULTS_DIR, "yourthings_cross_network_onboarding_v2.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
