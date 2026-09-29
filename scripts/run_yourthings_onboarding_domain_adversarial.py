"""Real test of the domain-adversarial fix (Section VII-G TODO) for the
cross-network generalization failure: does training the Phase 0 encoder
with an added domain-adversarial objective (federated_pretrain_encoder_domain_adversarial,
using each UNSW client's ID as a free domain label) improve on the
plain encoder's real result of 1/4 correct cross-network device matches?

Identical protocol to run_yourthings_onboarding_v2.py otherwise: same
UNSW data, same 15 pretraining rounds, same KB-building and query
procedure, same 4 real device-identity alignment pairs. Only the
encoder's training objective differs.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import numpy as np

from fedgate.data import load_unsw_device_identification, load_yourthings_device_identification
from fedgate.phase0_training import federated_pretrain_encoder_domain_adversarial, embed_all
from fedgate.onboarding import DeviceFingerprintKB, RuleBasedOnboardingPolicy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

YOURTHINGS_TO_UNSW_ALIGNMENT = {
    "SamsungSmartThingsHub": "Smart Things",
    "AmazonEchoGen1": "Amazon Echo",
    "BelkinWeMoMotionSensor": "Belkin wemo motion sensor",
    "LIFXVirtualBulb": "Light Bulbs LiFX Smart Bulb",
}


def main(phase0_rounds=15, grl_lambda=1.0, domain_loss_weight=1.0):
    print("Loading real UNSW data ...")
    unsw_clients = load_unsw_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=100,
        n_clients=10, seed=0,
    )
    unsw_classes = unsw_clients["unsw_client_0"]["classes"]
    print(f"UNSW: {len(unsw_classes)} device classes: {unsw_classes}")

    unsw_seqs = {cid: c["X"] for cid, c in unsw_clients.items()}
    unsw_labels = {cid: c["y_device"] for cid, c in unsw_clients.items()}

    print(f"\nTraining Phase 0 encoder federatedly on UNSW with domain-adversarial "
          f"objective ({phase0_rounds} rounds, grl_lambda={grl_lambda}, "
          f"domain_loss_weight={domain_loss_weight}) ...")
    t0 = time.time()
    encoder = federated_pretrain_encoder_domain_adversarial(
        unsw_seqs, seq_len=100, num_rounds=phase0_rounds, embed_dim=32,
        seed=0, grl_lambda=grl_lambda, domain_loss_weight=domain_loss_weight,
    )
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
    print(f"KB has {len(kb.entries)} admitted device fingerprints")

    print("\nLoading real YourThings data (IP-verified devices) ...")
    yt_clients = load_yourthings_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=5,
        n_clients=1, seed=0,
    )
    yt_classes = yt_clients["yourthings_client_0"]["classes"]
    yt_X = yt_clients["yourthings_client_0"]["X"]
    yt_y = yt_clients["yourthings_client_0"]["y_device"]

    print("\nEmbedding YourThings windows through the domain-adversarially-trained encoder ...")
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

    n_aligned = sum(1 for r in results if r["is_real_cross_network_same_device_test"])
    n_correct = sum(1 for r in results if r["correct_prediction_if_known"])
    n_auto_admit_unaligned = sum(
        1 for r in results if not r["is_real_cross_network_same_device_test"] and r["action"] == "auto-admit"
    )
    print(f"\n=== Summary: {n_correct}/{n_aligned} real cross-network matches correct "
          f"(plain encoder baseline: 1/4). False auto-admits among unaligned devices: "
          f"{n_auto_admit_unaligned} ===")

    out = {
        "description": "Domain-adversarial encoder: cross-dataset test, UNSW-trained "
                        "Phase0 encoder (masked-recon + domain-adversarial), real IP-verified YourThings traffic",
        "grl_lambda": grl_lambda,
        "domain_loss_weight": domain_loss_weight,
        "unsw_classes": unsw_classes,
        "yourthings_identified_devices": yt_classes,
        "alignment_used": YOURTHINGS_TO_UNSW_ALIGNMENT,
        "results": results,
        "n_real_cross_network_tests": n_aligned,
        "n_correct": n_correct,
        "n_false_auto_admit_unaligned": n_auto_admit_unaligned,
    }
    path = os.path.join(RESULTS_DIR, "yourthings_cross_network_domain_adversarial.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
