"""Real-LLM validation of onboarding and triage (README/main.tex open
item: "do the analogous real-LLM swap for onboarding.py and triage.py,
currently k-NN/threshold stand-ins, the same pattern the Gating Agent
itself started from"). Identical setup to
run_onboarding_and_triage.py (the rule-based version) -- same N-BaIoT
device-identity proxy for onboarding, same backbone classifier and
attack-signature KB for triage -- with LLMOnboardingPolicy /
LLMTriagePolicy (qwen2.5:14b) substituted for the rule-based stand-ins,
so results are directly comparable to results/onboarding_nbaiot_proxy.json
and results/triage_nbaiot.json.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import numpy as np
import torch

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation
from fedgate.onboarding import DeviceFingerprintKB, LLMOnboardingPolicy, LLMOnboardingPolicyError
from fedgate.triage import AttackSignatureKB, LLMTriagePolicy, LLMTriagePolicyError

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL = "qwen2.5:14b"


def run_onboarding(clients):
    print("=== LLM Onboarding validation (N-BaIoT devices as device-identity proxy) ===", flush=True)
    device_ids = list(clients.keys())
    held_out = device_ids[-1]
    admitted = device_ids[:-1]

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
        benign_mask = c.y_attack == 0
        centroid = standardize(c.X[benign_mask]).mean(axis=0)
        kb.add(label=d, embedding=centroid)

    policy = LLMOnboardingPolicy(model=MODEL, k=3)

    held_c = clients[held_out]
    held_benign = standardize(held_c.X[held_c.y_attack == 0])
    query_unseen = held_benign.mean(axis=0)
    action_unseen, rationale_unseen, conf_unseen, label_unseen = policy.decide(kb, query_unseen)
    print(f"unseen device query -> {action_unseen} ({label_unseen}, conf={conf_unseen:.2f}): {rationale_unseen}", flush=True)

    known_device = admitted[0]
    known_c = clients[known_device]
    known_benign = standardize(known_c.X[known_c.y_attack == 0])
    half = len(known_benign) // 2
    query_known = known_benign[half:].mean(axis=0)
    action_known, rationale_known, conf_known, label_known = policy.decide(kb, query_known)
    print(f"known device query -> {action_known} ({label_known}, conf={conf_known:.2f}): {rationale_known}", flush=True)

    out = {
        "model": MODEL,
        "admitted_devices": admitted,
        "held_out_device": held_out,
        "unseen_device_query": {
            "true_device": held_out, "action": action_unseen, "predicted_label": label_unseen,
            "confidence": conf_unseen, "rationale": rationale_unseen,
            "correct_behavior": action_unseen != "auto-admit",
        },
        "known_device_held_out_split_query": {
            "true_device": known_device, "action": action_known, "predicted_label": label_known,
            "confidence": conf_known, "rationale": rationale_known,
            "correct_behavior": action_known == "auto-admit" and label_known == known_device,
        },
    }
    with open(os.path.join(RESULTS_DIR, "onboarding_nbaiot_proxy_llm.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


def run_triage(clients, num_classes, embed_dim, num_rounds, n_sample_per_client=15):
    print("\n=== LLM Triage validation (on real backbone classifier's predictions) ===", flush=True)
    sim = FederatedSimulation(
        clients, num_classes=num_classes, embed_dim=embed_dim,
        malicious_clients=[], attack="none", gating=False, seed=0,
    )
    sim.run(num_rounds=num_rounds, log_every=num_rounds)

    class_names = list(sim.splits[sim.client_ids[0]][0].classes)

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

    policy = LLMTriagePolicy(model=MODEL, k=5)

    sim.global_model.eval()
    rows = []
    t0 = time.time()
    with torch.no_grad():
        for cid in sim.client_ids:
            _, _, test = sim.splits[cid]
            X = sim._standardize(test.X)
            logits = sim.global_model(X)
            preds = logits.argmax(dim=1).numpy()
            X_np = X.numpy()
            n_sample = min(n_sample_per_client, len(preds))
            rng = np.random.RandomState(0)
            sample_idx = rng.choice(len(preds), size=n_sample, replace=False)
            for i in sample_idx:
                pred_class = class_names[preds[i]]
                try:
                    action, rationale = policy.decide(kb, X_np[i], pred_class)
                except LLMTriagePolicyError as e:
                    action, rationale = "ERROR", str(e)
                rows.append({
                    "client": cid,
                    "true_class": class_names[test.y_attack[i]],
                    "predicted_class": pred_class,
                    "correct": bool(preds[i] == test.y_attack[i]),
                    "triage_action": action,
                    "rationale": rationale,
                })
            print(f"  {cid}: {n_sample} samples done ({time.time()-t0:.0f}s elapsed)", flush=True)

    import collections
    summary = collections.defaultdict(lambda: {"correct": 0, "incorrect": 0})
    for r in rows:
        if r["triage_action"] == "ERROR":
            continue
        key = r["triage_action"]
        summary[key]["correct" if r["correct"] else "incorrect"] += 1

    print("Triage action vs. actual prediction correctness:", flush=True)
    for action, counts in summary.items():
        total = counts["correct"] + counts["incorrect"]
        print(f"  {action:22s}: {counts['correct']}/{total} correct ({counts['correct']/total:.0%})", flush=True)

    out = {"model": MODEL, "rows": rows, "summary": {k: v for k, v in summary.items()}}
    with open(os.path.join(RESULTS_DIR, "triage_nbaiot_llm.json"), "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    print("Loading N-BaIoT ...", flush=True)
    clients_raw = load_nbaiot_federated(n_max_per_class=2000, seed=0)
    run_onboarding(clients_raw)
    run_triage(clients_raw, num_classes=3, embed_dim=115, num_rounds=15)
    print("\nDone.", flush=True)
