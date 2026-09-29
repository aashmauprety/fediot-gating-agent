"""Third domain-adversarial attempt (Section VII-G): v1 used per-UNSW-client
domains (no real network diversity -- null result, 1/4, same as plain
encoder); v2 added real YourThings devices as extra per-device domains
(real network diversity, but inconsistent granularity vs. UNSW's
per-client domains -- made things worse, 0/4). This version tests the
cleanest domain definition: exactly TWO domains, "UNSW network" and
"YourThings network," collapsing every UNSW client into domain 0 and
every added YourThings device into domain 1 -- a genuine, consistent
"which physical network" signal rather than per-client or per-device.

Same held-out evaluation set (4 devices, disjoint from pretraining) and
same pretraining-domain YourThings devices as v2, for a clean
apples-to-apples comparison across all three domain-adversarial variants
plus the plain encoder baseline.
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
HELD_OUT_EVAL_DEVICES = set(YOURTHINGS_TO_UNSW_ALIGNMENT.keys())
YOURTHINGS_PRETRAIN_DOMAIN_DEVICES = [
    "GoogleOnHub", "Sonos", "NestCamera", "RingDoorbell", "Roku4",
    "AmazonFireTV", "WithingsHome", "Wink2Hub",
]


def main(phase0_rounds=15, grl_lambda=1.0, domain_loss_weight=1.0):
    print("Loading real UNSW data ...")
    unsw_clients = load_unsw_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=100,
        n_clients=10, seed=0,
    )
    unsw_classes = unsw_clients["unsw_client_0"]["classes"]
    unsw_seqs = {cid: c["X"] for cid, c in unsw_clients.items()}
    unsw_labels = {cid: c["y_device"] for cid, c in unsw_clients.items()}

    print("\nLoading real YourThings data (IP-verified devices) ...")
    yt_clients = load_yourthings_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=5,
        n_clients=1, seed=0,
    )
    yt_classes = yt_clients["yourthings_client_0"]["classes"]
    yt_X_all = yt_clients["yourthings_client_0"]["X"]
    yt_y_all = yt_clients["yourthings_client_0"]["y_device"]

    pretrain_domain_seqs = {}
    for name in YOURTHINGS_PRETRAIN_DOMAIN_DEVICES:
        if name not in yt_classes:
            continue
        cls_idx = yt_classes.index(name)
        mask = yt_y_all == cls_idx
        if mask.sum() < 50:
            continue
        pretrain_domain_seqs[f"yourthings_{name}"] = yt_X_all[mask]

    combined_seqs = dict(unsw_seqs)
    combined_seqs.update(pretrain_domain_seqs)

    # Exactly two domains: 0 = UNSW network, 1 = YourThings network.
    domain_labels = {}
    for k in unsw_seqs:
        domain_labels[k] = 0
    for k in pretrain_domain_seqs:
        domain_labels[k] = 1
    print(f"\nDomains: {sum(1 for v in domain_labels.values() if v==0)} UNSW clients -> domain 0, "
          f"{sum(1 for v in domain_labels.values() if v==1)} YourThings devices -> domain 1 "
          f"(2 domains total, clean network-level signal)")

    print(f"\nTraining Phase 0 encoder federatedly with TWO-DOMAIN "
          f"(network-level) domain-adversarial objective ({phase0_rounds} rounds) ...")
    t0 = time.time()
    encoder = federated_pretrain_encoder_domain_adversarial(
        combined_seqs, seq_len=100, num_rounds=phase0_rounds, embed_dim=32,
        seed=0, grl_lambda=grl_lambda, domain_loss_weight=domain_loss_weight,
        domain_labels=domain_labels,
    )
    print(f"Done in {time.time()-t0:.1f}s")

    print("\nBuilding device-fingerprint KB from UNSW embeddings only ...")
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

    print("\nEmbedding all YourThings devices through the 2-domain-trained encoder ...")
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
            print(f"\n{cls_name} <-> UNSW '{aligned_unsw_class}' (HELD OUT from pretraining):")
            print(f"  action={action}, predicted={predicted_label}, confidence={confidence:.2%}, "
                  f"correct={row['correct_prediction_if_known']}")

    n_aligned = sum(1 for r in results if r["is_real_cross_network_same_device_test"])
    n_correct = sum(1 for r in results if r["correct_prediction_if_known"])
    print(f"\n=== Summary: {n_correct}/{n_aligned} real cross-network matches correct "
          f"(plain: 1/4; domain-adv v1 per-client: 1/4; domain-adv v2 per-device: 0/4) ===")

    out = {
        "description": "Domain-adversarial v3: exactly 2 domains (UNSW network vs. YourThings "
                        "network), evaluated on 4 held-out cross-network pairs",
        "grl_lambda": grl_lambda,
        "domain_loss_weight": domain_loss_weight,
        "domain_labels": domain_labels,
        "held_out_eval_devices": sorted(HELD_OUT_EVAL_DEVICES),
        "alignment_used": YOURTHINGS_TO_UNSW_ALIGNMENT,
        "results": results,
        "n_real_cross_network_tests": n_aligned,
        "n_correct": n_correct,
    }
    path = os.path.join(RESULTS_DIR, "yourthings_cross_network_domain_adversarial_v3.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
