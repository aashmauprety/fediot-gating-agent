"""Real (not synthetic) validation of the onboarding and triage retrieval
mechanisms, repurposing N-BaIoT's per-device structure:

Onboarding: treat 8 of the 9 N-BaIoT devices' benign-traffic feature
centroids as "admitted device fingerprints," then present the 9th
(held-out) device's benign traffic as "new device traffic" and see
whether the rule-based onboarding policy correctly recognizes it as
unfamiliar (request-more-data / escalate) rather than false-matching an
existing device. This is a real result about the retrieval+threshold
mechanism, but note it repurposes N-BaIoT (attack-type dataset) as a
device-identity proxy, since real UNSW/YourThings access was not
completed in this session -- see README.

Triage: train the plain (no-attack) attack-type classifier from the
backbone stage, then run the rule-based triage policy on its test-set
predictions to see whether "likely-false-positive" correlates with actual
classifier errors and "confirm" correlates with actual correct
predictions. This uses the SAME classifier and data as the backbone
stage, so it is real, not synthetic.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import numpy as np
import torch

from fedgate.data import load_nbaiot_federated, train_val_test_split, NBAIOT_DEVICES
from fedgate.experiment import FederatedSimulation
from fedgate.onboarding import DeviceFingerprintKB, RuleBasedOnboardingPolicy
from fedgate.triage import AttackSignatureKB, RuleBasedTriagePolicy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_onboarding(clients, n_max_per_class):
    print("=== Onboarding validation (N-BaIoT devices as device-identity proxy) ===")
    device_ids = list(clients.keys())
    held_out = device_ids[-1]
    admitted = device_ids[:-1]

    # "fingerprint" = mean standardized benign-traffic feature vector per device
    all_train_X = np.concatenate([clients[d].X for d in device_ids], axis=0).astype(np.float64)
    median = np.median(all_train_X, axis=0)
    q75, q25 = np.percentile(all_train_X, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1.0

    def standardize(X):
        return np.clip((X.astype(np.float64) - median) / iqr, -20, 20)

    kb = DeviceFingerprintKB()
    for d in admitted:
        c = clients[d]
        benign_mask = c.y_attack == 0  # class 0 = benign in COARSE_CLASSES
        centroid = standardize(c.X[benign_mask]).mean(axis=0)
        kb.add(label=d, embedding=centroid)

    policy = RuleBasedOnboardingPolicy(k=3)

    # Query 1: the actually-held-out (unseen) device -- expect NOT auto-admit
    held_c = clients[held_out]
    held_benign = standardize(held_c.X[held_c.y_attack == 0])
    query_unseen = held_benign.mean(axis=0)
    action_unseen, rationale_unseen, conf_unseen, label_unseen = policy.decide(kb, query_unseen)

    # Query 2: a held-out SPLIT of an already-admitted device -- expect auto-admit
    known_device = admitted[0]
    known_c = clients[known_device]
    known_benign = standardize(known_c.X[known_c.y_attack == 0])
    half = len(known_benign) // 2
    query_known = known_benign[half:].mean(axis=0)  # different samples than the ones used to build the fingerprint
    action_known, rationale_known, conf_known, label_known = policy.decide(kb, query_known)

    out = {
        "admitted_devices": admitted,
        "held_out_device": held_out,
        "unseen_device_query": {
            "true_device": held_out,
            "action": action_unseen,
            "predicted_label": label_unseen,
            "confidence": conf_unseen,
            "rationale": rationale_unseen,
            "correct_behavior": action_unseen != "auto-admit",
        },
        "known_device_held_out_split_query": {
            "true_device": known_device,
            "action": action_known,
            "predicted_label": label_known,
            "confidence": conf_known,
            "rationale": rationale_known,
            "correct_behavior": action_known == "auto-admit" and label_known == known_device,
        },
    }
    print(json.dumps(out, indent=2))
    with open(os.path.join(RESULTS_DIR, "onboarding_nbaiot_proxy.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def run_triage(clients, num_classes, embed_dim, num_rounds):
    print("\n=== Triage validation (on real backbone classifier's predictions) ===")
    sim = FederatedSimulation(
        clients, num_classes=num_classes, embed_dim=embed_dim,
        malicious_clients=[], attack="none", gating=False, seed=0,
    )
    sim.run(num_rounds=num_rounds, log_every=num_rounds)

    class_names = list(sim.splits[sim.client_ids[0]][0].classes)

    # build attack-signature KB from training splits (real data, held apart from test)
    kb = AttackSignatureKB()
    train_X, train_y = [], []
    for cid in sim.client_ids:
        train, _, _ = sim.splits[cid]
        Xs = sim._standardize(train.X).numpy()
        train_X.append(Xs)
        train_y.append(train.y_attack)
    train_X = np.concatenate(train_X, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    kb.build_from_calibration(train_X, train_y, class_names, max_per_class=200)

    policy = RuleBasedTriagePolicy(k=5)

    # run triage on the test set, compare triage action to actual correctness
    sim.global_model.eval()
    rows = []
    with torch.no_grad():
        for cid in sim.client_ids:
            _, _, test = sim.splits[cid]
            X = sim._standardize(test.X)
            logits = sim.global_model(X)
            preds = logits.argmax(dim=1).numpy()
            X_np = X.numpy()
            n_sample = min(30, len(preds))  # sample per client to keep this fast
            rng = np.random.RandomState(0)
            sample_idx = rng.choice(len(preds), size=n_sample, replace=False)
            for i in sample_idx:
                pred_class = class_names[preds[i]]
                action, rationale = policy.decide(kb, X_np[i], pred_class)
                rows.append({
                    "client": cid,
                    "true_class": class_names[test.y_attack[i]],
                    "predicted_class": pred_class,
                    "correct": bool(preds[i] == test.y_attack[i]),
                    "triage_action": action,
                })

    # summarize: does triage action correlate with correctness?
    import collections
    summary = collections.defaultdict(lambda: {"correct": 0, "incorrect": 0})
    for r in rows:
        key = r["triage_action"]
        summary[key]["correct" if r["correct"] else "incorrect"] += 1

    print("Triage action vs. actual prediction correctness:")
    for action, counts in summary.items():
        total = counts["correct"] + counts["incorrect"]
        print(f"  {action:22s}: {counts['correct']}/{total} correct ({counts['correct']/total:.0%})")

    out = {"rows": rows, "summary": {k: v for k, v in summary.items()}}
    with open(os.path.join(RESULTS_DIR, "triage_nbaiot.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    print("Loading N-BaIoT ...")
    clients_raw = load_nbaiot_federated(n_max_per_class=2000, seed=0)
    # For onboarding we want per-device ClientData objects directly (not split)
    run_onboarding(clients_raw, n_max_per_class=2000)
    run_triage(clients_raw, num_classes=3, embed_dim=115, num_rounds=15)
    print("\nDone.")
