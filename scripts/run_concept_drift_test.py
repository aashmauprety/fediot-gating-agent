"""Concept-drift adaptation test (Section VII limitation: "no
concept-drift adaptation"). Isolated from any poisoning attack --
`malicious_clients=[]`, every client is honest throughout -- so any
downweight/exclude decision here is a FALSE POSITIVE caused purely by
the synthetic benign concept drift injected into `calib_loss_delta`
for every client starting at round 10, not by any real attacker.

Compares the fixed-threshold `RuleBasedGatingPolicy` (calib_exclude_delta
absolute threshold) against `AdaptiveRuleBasedGatingPolicy` (EWMA
baseline, z-scored) on the identical drift, same data, same seed.
"""
import json
import os
import sys

sys.path.insert(0, "..")

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import RuleBasedGatingPolicy, AdaptiveRuleBasedGatingPolicy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "concept_drift_test.json")


def run_one(clients, policy, num_rounds, drift_start, drift_per_round, seed=0):
    sim = FederatedSimulation(
        clients, num_classes=3, embed_dim=115,
        malicious_clients=[],  # no attacker at all -- isolates drift-induced false positives
        attack="none", gating=True, seed=seed, gating_policy=policy,
        trust_decay=0.9, force_accept_round0=True,
        concept_drift_start_round=drift_start, concept_drift_per_round=drift_per_round,
    )
    logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    client_ids = list(clients.keys())

    # False-positive rate in the LAST 5 rounds (once drift has fully
    # accumulated): fraction of (honest, since there are no malicious
    # clients) decisions that were downweight/exclude.
    late_rounds = [h for h in sim.gating_history if h["round"] >= num_rounds - 5]
    n_total = sum(len(h["actions"]) for h in late_rounds)
    n_flagged = sum(
        1 for h in late_rounds for a in h["actions"].values() if a != "accept"
    )
    fp_rate_late = n_flagged / n_total if n_total else 0.0

    return {
        "final_macro_f1": logs[-1].macro_f1,
        "n_rounds": num_rounds,
        "false_positive_rate_last5rounds": fp_rate_late,
        "n_flagged_last5rounds": n_flagged,
        "n_total_last5rounds": n_total,
        "actions_by_round": {h["round"]: h["actions"] for h in sim.gating_history},
    }


def main(n_max_per_class=3000, num_rounds=30, drift_start=10, drift_per_round=0.03, seed=0):
    print(f"Loading N-BaIoT (n_max_per_class={n_max_per_class}) ...", flush=True)
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    print(f"{len(clients)} clients, all honest. Synthetic drift starts round {drift_start}, "
          f"+{drift_per_round}/round (reaching +{(num_rounds - drift_start) * drift_per_round:.2f} "
          f"by the final round).", flush=True)

    print("\n=== Fixed-threshold RuleBasedGatingPolicy ===", flush=True)
    fixed_result = run_one(
        clients, RuleBasedGatingPolicy(warmup_rounds=5, warmup_scale=1.5),
        num_rounds, drift_start, drift_per_round, seed,
    )
    print(f"F1={fixed_result['final_macro_f1']:.4f}, "
          f"false-positive rate (last 5 rounds) = {fixed_result['false_positive_rate_last5rounds']:.1%} "
          f"({fixed_result['n_flagged_last5rounds']}/{fixed_result['n_total_last5rounds']})", flush=True)

    print("\n=== AdaptiveRuleBasedGatingPolicy (same-round cross-client z-score) ===", flush=True)
    adaptive_result = run_one(
        clients, AdaptiveRuleBasedGatingPolicy(warmup_rounds=5, warmup_scale=1.5),
        num_rounds, drift_start, drift_per_round, seed,
    )
    print(f"F1={adaptive_result['final_macro_f1']:.4f}, "
          f"false-positive rate (last 5 rounds) = {adaptive_result['false_positive_rate_last5rounds']:.1%} "
          f"({adaptive_result['n_flagged_last5rounds']}/{adaptive_result['n_total_last5rounds']})", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump({
            "description": (
                "Synthetic benign concept-drift test, NO attacker "
                "(malicious_clients=[]) -- isolates drift-induced false "
                "positives from real poisoning. Fixed-threshold vs "
                "adaptive-EWMA-baseline RuleBasedGatingPolicy, identical "
                "data/seed/drift."
            ),
            "drift_start_round": drift_start,
            "drift_per_round": drift_per_round,
            "num_rounds": num_rounds,
            "fixed_threshold": fixed_result,
            "adaptive_baseline": adaptive_result,
        }, f, indent=2)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
