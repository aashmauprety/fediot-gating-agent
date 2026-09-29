"""Second, corrected attempt at the domain-adversarial fix for
cross-network generalization (Section VII-G). The first attempt
(run_yourthings_onboarding_domain_adversarial.py) used UNSW's 10
federated clients as the only "domains" -- but those are a simulated
partition of ONE real network, not 10 different networks, so the
domain-adversarial signal never saw genuine network diversity, which we
diagnosed as why it made no difference (still 1/4 correct).

This version fixes that: the encoder is pretrained on UNSW's 10 clients
PLUS several *other* real YourThings devices (unlabeled, used only as
extra domains for the adversarial signal) -- giving the discriminator
genuine cross-network diversity to learn from. Critically, the YourThings
devices used for pretraining are DISJOINT from the 4 held-out evaluation
devices (Samsung SmartThings Hub, Amazon Echo, Belkin WeMo Motion Sensor,
LiFX Smart Bulb) -- no eval data leaks into pretraining. The evaluation
protocol is otherwise identical to the previous two runs, so the three
results (plain encoder, domain-adversarial v1, domain-adversarial v2)
are a real, apples-to-apples comparison.
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

# Held out from pretraining entirely -- evaluation-only.
HELD_OUT_EVAL_DEVICES = set(YOURTHINGS_TO_UNSW_ALIGNMENT.keys())

# A handful of other real, active YourThings devices used purely to give
# the domain-adversarial objective genuine second-network traffic during
# pretraining -- picked for reasonable packet counts, not cherry-picked
# for any outcome-related property.
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
    print(f"UNSW: {len(unsw_classes)} device classes, {len(unsw_seqs)} clients")

    print("\nLoading real YourThings data (IP-verified devices) for BOTH "
          "pretraining-domain-diversity and held-out evaluation ...")
    yt_clients = load_yourthings_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=5,
        n_clients=1, seed=0,
    )
    yt_classes = yt_clients["yourthings_client_0"]["classes"]
    yt_X_all = yt_clients["yourthings_client_0"]["X"]
    yt_y_all = yt_clients["yourthings_client_0"]["y_device"]

    # Split YourThings devices into pretraining-domain-source vs.
    # held-out evaluation, strictly disjoint.
    pretrain_domain_seqs = {}
    for name in YOURTHINGS_PRETRAIN_DOMAIN_DEVICES:
        if name not in yt_classes:
            print(f"  (skipping {name}: not active in this capture)")
            continue
        cls_idx = yt_classes.index(name)
        mask = yt_y_all == cls_idx
        if mask.sum() < 50:
            print(f"  (skipping {name}: only {mask.sum()} windows)")
            continue
        pretrain_domain_seqs[f"yourthings_{name}"] = yt_X_all[mask]
        print(f"  using {name} as pretraining domain: {mask.sum()} windows")

    combined_seqs = dict(unsw_seqs)
    combined_seqs.update(pretrain_domain_seqs)
    print(f"\nCombined pretraining set: {len(unsw_seqs)} UNSW clients + "
          f"{len(pretrain_domain_seqs)} real YourThings domains = {len(combined_seqs)} total domains")

    print(f"\nTraining Phase 0 encoder federatedly with GENUINE multi-network "
          f"domain-adversarial objective ({phase0_rounds} rounds) ...")
    t0 = time.time()
    encoder = federated_pretrain_encoder_domain_adversarial(
        combined_seqs, seq_len=100, num_rounds=phase0_rounds, embed_dim=32,
        seed=0, grl_lambda=grl_lambda, domain_loss_weight=domain_loss_weight,
    )
    print(f"Done in {time.time()-t0:.1f}s")

    print("\nBuilding device-fingerprint KB from UNSW embeddings only (unchanged) ...")
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

    print("\nEmbedding ALL YourThings devices (including held-out eval set) "
          "through the multi-network-trained encoder ...")
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
        used_in_pretraining = f"yourthings_{cls_name}" in pretrain_domain_seqs
        row = {
            "yourthings_device": cls_name,
            "n_windows": int(mask.sum()),
            "aligned_unsw_class": aligned_unsw_class,
            "is_real_cross_network_same_device_test": aligned_unsw_class is not None,
            "used_as_pretraining_domain": used_in_pretraining,
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
          f"(plain encoder: 1/4; domain-adversarial v1 (UNSW-only domains): 1/4) ===")

    out = {
        "description": "Domain-adversarial v2: encoder pretrained on UNSW + real disjoint YourThings "
                        "devices (genuine multi-network domains), evaluated on 4 held-out cross-network pairs",
        "grl_lambda": grl_lambda,
        "domain_loss_weight": domain_loss_weight,
        "pretraining_domains": list(combined_seqs.keys()),
        "held_out_eval_devices": sorted(HELD_OUT_EVAL_DEVICES),
        "alignment_used": YOURTHINGS_TO_UNSW_ALIGNMENT,
        "results": results,
        "n_real_cross_network_tests": n_aligned,
        "n_correct": n_correct,
    }
    path = os.path.join(RESULTS_DIR, "yourthings_cross_network_domain_adversarial_v2.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
