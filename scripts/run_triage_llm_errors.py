"""Follow-up to run_onboarding_and_triage_llm.py: that run's 135-sample
draw (15/client) happened to contain zero classifier errors (backbone is
near-perfect, consistent with the rule-based triage validation's own
note that only 1/270 stratified samples was a real error). Testing
whether LLMTriagePolicy can actually catch a real error requires
deliberately finding one, not hoping a random sample includes it. This
scans the FULL test set across all clients for actual misclassifications
and runs the LLM triage policy specifically on those, plus a small
matched sample of correct predictions for contrast.
"""
import json
import os
import sys

sys.path.insert(0, "..")

import numpy as np
import torch

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation
from fedgate.triage import AttackSignatureKB, LLMTriagePolicy, LLMTriagePolicyError

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL = "qwen2.5:14b"


def main(num_rounds=15, n_correct_sample=10):
    print("Loading N-BaIoT ...", flush=True)
    clients = load_nbaiot_federated(n_max_per_class=2000, seed=0)
    sim = FederatedSimulation(
        clients, num_classes=3, embed_dim=115,
        malicious_clients=[], attack="none", gating=False, seed=0,
    )
    sim.run(num_rounds=num_rounds, log_every=num_rounds)
    class_names = list(sim.splits[sim.client_ids[0]][0].classes)

    kb = AttackSignatureKB()
    train_X, train_y = [], []
    for cid in sim.client_ids:
        train, _, _ = sim.splits[cid]
        train_X.append(sim._standardize(train.X).numpy())
        train_y.append(train.y_attack)
    train_X = np.concatenate(train_X, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    kb.build_from_calibration(train_X, train_y, class_names, max_per_class=200)

    # scan full test set for real misclassifications
    sim.global_model.eval()
    errors, corrects = [], []
    with torch.no_grad():
        for cid in sim.client_ids:
            _, _, test = sim.splits[cid]
            X = sim._standardize(test.X)
            logits = sim.global_model(X)
            preds = logits.argmax(dim=1).numpy()
            X_np = X.numpy()
            for i in range(len(preds)):
                row = (cid, X_np[i], class_names[preds[i]], class_names[test.y_attack[i]], bool(preds[i] == test.y_attack[i]))
                (errors if not row[4] else corrects).append(row)
    print(f"Full test set: {len(errors)} real misclassifications out of {len(errors)+len(corrects)} total.", flush=True)

    rng = np.random.RandomState(0)
    correct_sample = [corrects[i] for i in rng.choice(len(corrects), size=min(n_correct_sample, len(corrects)), replace=False)]
    to_run = errors + correct_sample
    print(f"Running LLM triage on all {len(errors)} real errors + {len(correct_sample)} correct samples for contrast.", flush=True)

    policy = LLMTriagePolicy(model=MODEL, k=5)
    rows = []
    for cid, x, pred_class, true_class, correct in to_run:
        try:
            action, rationale = policy.decide(kb, x, pred_class)
        except LLMTriagePolicyError as e:
            action, rationale = "ERROR", str(e)
        rows.append({
            "client": cid, "predicted_class": pred_class, "true_class": true_class,
            "correct": correct, "triage_action": action, "rationale": rationale,
        })
        print(f"  [{'CORRECT' if correct else 'ERROR  '}] pred={pred_class} true={true_class} -> {action} | {rationale[:100]}", flush=True)

    n_errors_flagged = sum(1 for r in rows if not r["correct"] and r["triage_action"] == "likely-false-positive")
    n_correct_confirmed = sum(1 for r in rows if r["correct"] and r["triage_action"] == "confirm")
    print(f"\nOf {len(errors)} real classifier errors, {n_errors_flagged} were flagged likely-false-positive by the LLM.", flush=True)
    print(f"Of {len(correct_sample)} correct predictions sampled, {n_correct_confirmed} were confirmed by the LLM.", flush=True)

    out = {
        "model": MODEL,
        "n_total_test_samples": len(errors) + len(corrects),
        "n_real_errors": len(errors),
        "n_errors_flagged_likely_false_positive": n_errors_flagged,
        "n_correct_sampled": len(correct_sample),
        "n_correct_confirmed": n_correct_confirmed,
        "rows": rows,
    }
    with open(os.path.join(RESULTS_DIR, "triage_nbaiot_llm_errors.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {os.path.join(RESULTS_DIR, 'triage_nbaiot_llm_errors.json')}", flush=True)


if __name__ == "__main__":
    main()
