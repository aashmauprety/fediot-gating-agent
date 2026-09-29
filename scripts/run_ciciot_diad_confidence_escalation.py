"""Real test of confidence-gated LLM escalation (the design the user
actually asked for): train the real CIC IoT-DIAD 2024 device-ID
classifier (59 real devices), split its test predictions into
"confident" (top-1 probability >= threshold) and "uncertain" (below),
and on the uncertain subset specifically, compare three policies:
  (a) classifier's own top-1 guess (the do-nothing baseline)
  (b) rule-based escalation (same top-1, just relabels very-low-confidence
      cases as "escalate_to_human" instead of guessing -- no real
      reasoning)
  (c) real LLM escalation (Ollama-backed, given the classifier's own
      probability distribution over its top candidates, asked to reason
      about which is actually correct or whether to escalate)

This isolates whether the LLM adds real value exactly where it's
supposed to matter -- the uncertain cases -- rather than being
diluted by the confident majority, where any reasonable policy just
matches the classifier anyway.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import numpy as np

from fedgate.data import load_ciciot_diad_device_identification
from fedgate.classifier_training import federated_train_classifier
from fedgate.confidence_escalation import LLMConfidenceEscalationPolicy, ConfidenceEscalationError, rule_based_escalation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def robust_scale(client_X: dict) -> dict:
    all_X = np.concatenate(list(client_X.values()), axis=0).astype(np.float64)
    median = np.median(all_X, axis=0)
    q75, q25 = np.percentile(all_X, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1.0
    return {
        cid: np.clip((X.astype(np.float64) - median) / iqr, -20.0, 20.0).astype(np.float32)
        for cid, X in client_X.items()
    }


def main(n_max_per_device=2000, num_rounds=25, n_clients=10, confidence_threshold=0.15,
         max_llm_calls=150, seed=0):
    print("Loading CIC IoT-DIAD 2024 device-ID data ...")
    clients = load_ciciot_diad_device_identification(
        n_max_per_device=n_max_per_device, min_rows_per_device=50, n_clients=n_clients, seed=seed,
    )
    classes = list(clients.values())[0]["classes"]
    print(f"{len(classes)} device classes, {len(clients)} clients")

    client_X = {cid: c["X"] for cid, c in clients.items()}
    client_X = robust_scale(client_X)
    client_y = {cid: c["y_device"] for cid, c in clients.items()}

    print(f"\nTraining federated classifier ({num_rounds} rounds) ...")
    t0 = time.time()
    result = federated_train_classifier(
        client_X, client_y, num_classes=len(classes), embed_dim=client_X[list(client_X)[0]].shape[1],
        num_rounds=num_rounds, seed=seed, return_model=True,
    )
    print(f"Classifier trained in {time.time()-t0:.1f}s, baseline macro-F1={result['final_macro_f1']:.4f}")

    probs = result["final_probs"]  # (n_test, n_classes)
    y_true = np.array(result["final_true"])
    y_pred_top1 = np.array(result["final_preds"])
    top1_conf = probs.max(axis=1)

    confident_mask = top1_conf >= confidence_threshold
    uncertain_idx = np.where(~confident_mask)[0]
    print(f"\n{confident_mask.sum()}/{len(y_true)} samples confident (>= {confidence_threshold:.0%}), "
          f"{len(uncertain_idx)} uncertain")

    if len(uncertain_idx) > max_llm_calls:
        rng = np.random.RandomState(seed)
        uncertain_idx = rng.choice(uncertain_idx, size=max_llm_calls, replace=False)
        print(f"Subsampled to {max_llm_calls} uncertain cases for the real LLM comparison "
              f"(wall-clock cost of real Ollama calls)")

    # (a) classifier top-1 baseline, on the uncertain subset
    baseline_correct = (y_pred_top1[uncertain_idx] == y_true[uncertain_idx]).sum()

    # (b) rule-based escalation, on the uncertain subset
    rb_correct = 0
    rb_escalated = 0
    for i in uncertain_idx:
        d = rule_based_escalation(classes, probs[i])
        if d.action == "escalate_to_human":
            rb_escalated += 1
        elif classes.index(d.predicted_label) == y_true[i]:
            rb_correct += 1

    # (c) real LLM escalation, on the uncertain subset
    print(f"\nRunning real LLM escalation on {len(uncertain_idx)} uncertain cases ...")
    llm_policy = LLMConfidenceEscalationPolicy(model="llama3.1:8b")
    llm_correct = 0
    llm_escalated = 0
    llm_errors = 0
    per_case = []
    t0 = time.time()
    for n, i in enumerate(uncertain_idx):
        try:
            d = llm_policy.decide(classes, probs[i])
        except ConfidenceEscalationError as e:
            llm_errors += 1
            continue
        is_correct = (d.predicted_label == classes[y_true[i]]) if d.action != "escalate_to_human" else None
        if d.action == "escalate_to_human":
            llm_escalated += 1
        elif is_correct:
            llm_correct += 1
        per_case.append({
            "true_label": classes[y_true[i]],
            "classifier_top1": classes[y_pred_top1[i]],
            "top1_confidence": float(top1_conf[i]),
            "llm_predicted": d.predicted_label,
            "llm_action": d.action,
            "llm_rationale": d.rationale,
            "correct": is_correct,
        })
        if (n + 1) % 25 == 0:
            print(f"  {n+1}/{len(uncertain_idx)} done ({time.time()-t0:.0f}s elapsed)")
    elapsed = time.time() - t0

    n_eval = len(uncertain_idx)
    print(f"\n=== Results on {n_eval} uncertain (low-confidence) test cases ===")
    print(f"  (a) Classifier top-1 baseline:   {baseline_correct}/{n_eval} = {baseline_correct/n_eval:.1%}")
    print(f"  (b) Rule-based escalation:       {rb_correct}/{n_eval} correct, {rb_escalated} escalated "
          f"({rb_correct/n_eval:.1%} correct of all, {rb_correct/(n_eval-rb_escalated) if n_eval>rb_escalated else 0:.1%} of non-escalated)")
    print(f"  (c) Real LLM escalation:         {llm_correct}/{n_eval} correct, {llm_escalated} escalated, "
          f"{llm_errors} errors ({llm_correct/n_eval:.1%} correct of all, "
          f"{llm_correct/(n_eval-llm_escalated) if n_eval>llm_escalated else 0:.1%} of non-escalated) ({elapsed:.0f}s)")

    out = {
        "description": "Real confidence-gated escalation test, CIC IoT-DIAD 2024 device-ID (59 classes)",
        "n_classes": len(classes),
        "baseline_macro_f1": result["final_macro_f1"],
        "confidence_threshold": confidence_threshold,
        "n_confident": int(confident_mask.sum()),
        "n_uncertain_total": int(len(np.where(~confident_mask)[0])),
        "n_uncertain_evaluated": n_eval,
        "classifier_top1_correct": int(baseline_correct),
        "rule_based_correct": int(rb_correct),
        "rule_based_escalated": int(rb_escalated),
        "llm_correct": int(llm_correct),
        "llm_escalated": int(llm_escalated),
        "llm_errors": int(llm_errors),
        "llm_wall_clock_seconds": elapsed,
        "per_case": per_case,
    }
    path = os.path.join(RESULTS_DIR, "ciciot_diad_confidence_escalation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
