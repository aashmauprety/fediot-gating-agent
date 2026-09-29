"""Real test of contrastive cross-network alignment (Section VII-G's
most promising untried direction after three domain-adversarial
configurations all failed to beat the plain encoder's 1/4 real
cross-network matches). Uses real cross-network device-identity
supervision -- the same 4 verified alignment pairs used throughout this
investigation (Samsung SmartThings Hub, Amazon Echo, Belkin WeMo Motion
Sensor, LiFX Smart Bulb) -- to explicitly pull each pair's embeddings
together via a supervised contrastive (InfoNCE) loss.

HONESTY-CRITICAL METHODOLOGY: aligning on all 4 known pairs and then
testing whether those same 4 pairs match afterwards would not be a real
test -- it would just be checking whether the model memorized the exact
examples it was shown. This runs a real 4-fold leave-one-out: for each
of the 4 pairs, align using the OTHER 3, then evaluate cross-network
retrieval on the held-out 4th, which the fine-tuning never saw. This is
the only way to honestly answer "does explicit alignment generalize to
an unseen device pair," which is what the technique would need to do to
be useful (the whole point is admitting NEW device types you don't
already have a labeled pair for).
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import numpy as np

from fedgate.data import load_unsw_device_identification, load_yourthings_device_identification
from fedgate.phase0_training import federated_pretrain_encoder, contrastive_finetune_encoder, embed_all
from fedgate.onboarding import DeviceFingerprintKB, RuleBasedOnboardingPolicy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

YOURTHINGS_TO_UNSW_ALIGNMENT = {
    "SamsungSmartThingsHub": "Smart Things",
    "AmazonEchoGen1": "Amazon Echo",
    "BelkinWeMoMotionSensor": "Belkin wemo motion sensor",
    "LIFXVirtualBulb": "Light Bulbs LiFX Smart Bulb",
}


def main(phase0_rounds=15, num_finetune_steps=300):
    print("Loading real UNSW data ...")
    unsw_clients = load_unsw_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=100,
        n_clients=10, seed=0,
    )
    unsw_classes = unsw_clients["unsw_client_0"]["classes"]
    unsw_seqs = {cid: c["X"] for cid, c in unsw_clients.items()}
    unsw_labels = {cid: c["y_device"] for cid, c in unsw_clients.items()}

    # Pool all UNSW windows by class, across clients -- needed for the
    # per-class positive/negative sampling the contrastive loss uses.
    unsw_by_class = {c: [] for c in unsw_classes}
    for cid in unsw_clients:
        X = unsw_seqs[cid]
        y = unsw_labels[cid]
        for i, lbl in enumerate(y):
            unsw_by_class[unsw_classes[lbl]].append(X[i])
    unsw_by_class = {c: np.stack(v) for c, v in unsw_by_class.items() if v}

    print(f"\nPretraining plain (non-adversarial) Phase 0 encoder on UNSW "
          f"({phase0_rounds} rounds) -- same baseline as the plain-encoder result ...")
    t0 = time.time()
    base_encoder = federated_pretrain_encoder(unsw_seqs, seq_len=100, num_rounds=phase0_rounds, embed_dim=32, seed=0)
    print(f"Done in {time.time()-t0:.1f}s")

    print("\nLoading real YourThings data (IP-verified devices) ...")
    yt_clients = load_yourthings_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=5,
        n_clients=1, seed=0,
    )
    yt_classes = yt_clients["yourthings_client_0"]["classes"]
    yt_X_all = yt_clients["yourthings_client_0"]["X"]
    yt_y_all = yt_clients["yourthings_client_0"]["y_device"]
    yt_by_device = {}
    for name in YOURTHINGS_TO_UNSW_ALIGNMENT:
        if name not in yt_classes:
            continue
        idx = yt_classes.index(name)
        yt_by_device[name] = yt_X_all[yt_y_all == idx]

    policy = RuleBasedOnboardingPolicy(k=5)
    fold_results = []

    for held_out_yt, held_out_unsw in YOURTHINGS_TO_UNSW_ALIGNMENT.items():
        print(f"\n{'='*70}\nFOLD: holding out {held_out_yt} <-> {held_out_unsw}\n{'='*70}")

        # Build positive pairs from the OTHER 3 known alignments only --
        # the held-out pair is never shown to the fine-tuning step.
        positive_pairs = {}
        for yt_name, unsw_cls in YOURTHINGS_TO_UNSW_ALIGNMENT.items():
            if yt_name == held_out_yt:
                continue
            if yt_name not in yt_by_device:
                continue
            positive_pairs[unsw_cls] = (unsw_by_class[unsw_cls], yt_by_device[yt_name])
        print(f"  Aligning on {len(positive_pairs)} pairs: {list(positive_pairs.keys())}")

        t0 = time.time()
        finetuned = contrastive_finetune_encoder(
            base_encoder, positive_pairs, unsw_by_class,
            num_steps=num_finetune_steps, seed=0,
        )
        print(f"  Contrastive fine-tuning done in {time.time()-t0:.1f}s")

        # Rebuild KB with the fine-tuned encoder.
        unsw_emb = embed_all(finetuned, unsw_seqs)
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

        # Query with the HELD-OUT device only -- this is the real test.
        held_out_X = yt_by_device[held_out_yt]
        held_out_emb = embed_all(finetuned, {"yt": held_out_X})["yt"]
        query_embedding = held_out_emb.mean(axis=0)
        action, rationale, confidence, predicted_label = policy.decide(kb, query_embedding)
        correct = predicted_label == held_out_unsw

        print(f"  HELD-OUT RESULT: {held_out_yt} -> predicted={predicted_label}, "
              f"confidence={confidence:.2%}, correct={correct}")

        fold_results.append({
            "held_out_yourthings_device": held_out_yt,
            "held_out_unsw_class": held_out_unsw,
            "n_alignment_pairs_used": len(positive_pairs),
            "action": action,
            "predicted_label": predicted_label,
            "confidence": confidence,
            "correct": correct,
        })

    n_correct = sum(1 for r in fold_results if r["correct"])
    print(f"\n{'='*70}\n=== FINAL: {n_correct}/4 held-out folds correct "
          f"(plain: 1/4; domain-adv v1/v3: 1/4; domain-adv v2: 0/4) ===\n{'='*70}")

    out = {
        "description": "Real 4-fold leave-one-out contrastive cross-network alignment: "
                        "align on 3 known pairs, test on the unseen 4th, repeated for each",
        "num_finetune_steps": num_finetune_steps,
        "fold_results": fold_results,
        "n_correct": n_correct,
        "n_folds": 4,
    }
    path = os.path.join(RESULTS_DIR, "yourthings_cross_network_contrastive_alignment.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
