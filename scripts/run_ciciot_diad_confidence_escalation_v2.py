"""Follow-up to run_ciciot_diad_confidence_escalation.py: that experiment
found LLM escalation using ONLY the classifier's probability distribution
does not beat the classifier's raw top-1 guess (18.0% vs 18.7% on the
hardest cases), and actively hurts when the LLM disagrees with the
classifier (1/9 correct on overrides). Diagnosis: the probabilities are
a strict derivative of what the classifier already extracted from the
features, so the LLM had no genuinely new evidence to reason from.

This version gives the LLM real, named, raw (unscaled) feature values
for each sample too -- ports, protocol flags, TTL, TCP window size,
payload entropy/length, timing -- alongside the same probability
distribution, so it has actual independent evidence (the kind a person
with protocol knowledge could reason about) rather than just a
repackaging of the classifier's own output.

Same real protocol as v1: same classifier, same confidence threshold,
same uncertain-subset evaluation, so the two are directly comparable.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import numpy as np

from fedgate.data import load_ciciot_diad_device_identification
from fedgate.classifier_training import federated_train_classifier
from fedgate.confidence_escalation import (
    LLMFeatureAwareEscalationPolicy, ConfidenceEscalationError, DIAD_INTERPRETABLE_FEATURES,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def robust_scale_params(all_X: np.ndarray):
    median = np.median(all_X, axis=0)
    q75, q25 = np.percentile(all_X, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1.0
    return median, iqr


def main(n_max_per_device=2000, num_rounds=25, n_clients=10, confidence_threshold=0.15,
         max_llm_calls=150, seed=0):
    print("Loading CIC IoT-DIAD 2024 device-ID data ...")
    clients = load_ciciot_diad_device_identification(
        n_max_per_device=n_max_per_device, min_rows_per_device=50, n_clients=n_clients, seed=seed,
    )
    classes = list(clients.values())[0]["classes"]
    all_feature_cols = None
    from fedgate.data import load_ciciot_diad_numeric_feature_columns, CICIOT_DIAD_DEVICE_TASK
    all_feature_cols = load_ciciot_diad_numeric_feature_columns(CICIOT_DIAD_DEVICE_TASK)
    interp_idx = [all_feature_cols.index(f) for f in DIAD_INTERPRETABLE_FEATURES]

    raw_client_X = {cid: c["X"] for cid, c in clients.items()}
    all_raw_X = np.concatenate(list(raw_client_X.values()), axis=0).astype(np.float64)
    median, iqr = robust_scale_params(all_raw_X)
    client_X = {
        cid: np.clip((X.astype(np.float64) - median) / iqr, -20.0, 20.0).astype(np.float32)
        for cid, X in raw_client_X.items()
    }
    client_y = {cid: c["y_device"] for cid, c in clients.items()}

    print(f"\nTraining federated classifier ({num_rounds} rounds) ...")
    t0 = time.time()
    result = federated_train_classifier(
        client_X, client_y, num_classes=len(classes), embed_dim=client_X[list(client_X)[0]].shape[1],
        num_rounds=num_rounds, seed=seed, return_model=True,
    )
    print(f"Classifier trained in {time.time()-t0:.1f}s, baseline macro-F1={result['final_macro_f1']:.4f}")

    probs = result["final_probs"]
    y_true = np.array(result["final_true"])
    y_pred_top1 = np.array(result["final_preds"])
    scaled_test_X = result["final_test_X"]
    # Invert the robust scaling to recover real, interpretable raw values
    # for display to the LLM (affine inverse; minor distortion only for
    # the rare values that hit the +-20 clip boundary).
    raw_test_X = scaled_test_X.astype(np.float64) * iqr + median
    top1_conf = probs.max(axis=1)

    confident_mask = top1_conf >= confidence_threshold
    uncertain_idx = np.where(~confident_mask)[0]
    print(f"\n{confident_mask.sum()}/{len(y_true)} samples confident (>= {confidence_threshold:.0%}), "
          f"{len(uncertain_idx)} uncertain")

    if len(uncertain_idx) > max_llm_calls:
        rng = np.random.RandomState(seed)
        uncertain_idx = rng.choice(uncertain_idx, size=max_llm_calls, replace=False)
        print(f"Subsampled to {max_llm_calls} uncertain cases (same seed as v1, for comparability)")

    baseline_correct = (y_pred_top1[uncertain_idx] == y_true[uncertain_idx]).sum()

    print(f"\nRunning real feature-aware LLM escalation on {len(uncertain_idx)} uncertain cases ...")
    llm_policy = LLMFeatureAwareEscalationPolicy(model="llama3.1:8b")
    llm_correct = 0
    llm_escalated = 0
    llm_errors = 0
    n_overrides = 0
    n_overrides_correct = 0
    per_case = []
    t0 = time.time()
    for n, i in enumerate(uncertain_idx):
        feat_vals = raw_test_X[i, interp_idx]
        try:
            d = llm_policy.decide(classes, probs[i], DIAD_INTERPRETABLE_FEATURES, feat_vals)
        except ConfidenceEscalationError:
            llm_errors += 1
            continue
        is_correct = (d.predicted_label == classes[y_true[i]]) if d.action != "escalate_to_human" else None
        if d.action == "escalate_to_human":
            llm_escalated += 1
        elif is_correct:
            llm_correct += 1
        if d.action == "override":
            n_overrides += 1
            if is_correct:
                n_overrides_correct += 1
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
    print(f"\n=== Results on {n_eval} uncertain cases, WITH raw feature evidence ===")
    print(f"  Classifier top-1 baseline:        {baseline_correct}/{n_eval} = {baseline_correct/n_eval:.1%}")
    print(f"  Feature-aware LLM escalation:      {llm_correct}/{n_eval} correct, {llm_escalated} escalated, "
          f"{llm_errors} errors = {llm_correct/n_eval:.1%} ({elapsed:.0f}s)")
    print(f"  LLM overrides: {n_overrides}, correct: {n_overrides_correct} "
          f"({n_overrides_correct/n_overrides if n_overrides else 0:.1%})")
    print(f"  Reference (v1, probability-only): baseline 18.7%, LLM 18.0%, overrides 1/9 correct (11%)")

    out = {
        "description": "Real confidence-gated escalation, WITH raw feature evidence, CIC IoT-DIAD 2024 device-ID",
        "n_classes": len(classes),
        "baseline_macro_f1": result["final_macro_f1"],
        "confidence_threshold": confidence_threshold,
        "n_uncertain_evaluated": n_eval,
        "classifier_top1_correct": int(baseline_correct),
        "llm_correct": int(llm_correct),
        "llm_escalated": int(llm_escalated),
        "llm_errors": int(llm_errors),
        "n_overrides": n_overrides,
        "n_overrides_correct": n_overrides_correct,
        "llm_wall_clock_seconds": elapsed,
        "features_shown": DIAD_INTERPRETABLE_FEATURES,
        "per_case": per_case,
    }
    path = os.path.join(RESULTS_DIR, "ciciot_diad_confidence_escalation_v2_features.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
