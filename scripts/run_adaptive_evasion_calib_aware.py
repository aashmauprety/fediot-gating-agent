"""Calibration-preserving adaptive attacker (review-response follow-up,
Oct 2026 deadline extension): every adaptive-evasion variant tried so
far (`evasion_lambda`, `evasion_collude`) blends a poisoned update's
DIRECTION toward a clean reference to evade the update-norm and cosine
digest signals, but none of them were ever tested against the THIRD
digest signal, calib_loss_delta, at all -- an attacker could in
principle find a blend that looks directionally honest while still
measurably increasing the global model's held-out loss.

`FederatedSimulation(evasion_calib_aware=True)` grid-searches blend
strength per malicious client per round (see
`fedgate/experiment.py:_select_calib_aware_lambda`), picking the
smallest lambda (= strongest residual attack) whose calibration impact
-- measured on the attacker's OWN held-out val split, a genuinely
locally-available proxy for the server's real calib_loss_delta --
stays within `calib_aware_safe_delta` (default 0.5, the same absolute
threshold `RuleBasedGatingPolicy` uses, assumed known to the attacker
as a conservative worst-case, consistent with the constrain-and-scale
literature already cited in the threat model). It does NOT assume the
attacker can see other clients' calib deltas this round, so it cannot
directly target the real policy's same-round cross-client z-score
(`AdaptiveRuleBasedGatingPolicy`) -- only its own absolute impact.

Two defenses tested, both deterministic (cheap, no LLM calls needed for
this sweep): GShield-style clustering (a control -- it never looks at
calib_loss_delta at all, so this attacker variant should change nothing
for it) and `AdaptiveRuleBasedGatingPolicy` (the same-round z-score
rule-based reference the paper's real LLM Gating Agent is benchmarked
against). N-BaIoT, targeted attack (the attack type most sensitive to
a defense actually failing, since success rate is a clean pass/fail
signal independent of aggregate accuracy).
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import AdaptiveRuleBasedGatingPolicy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "adaptive_evasion_calib_aware.json")


def run_gshield(clients, num_classes, embed_dim, malicious, lam, calib_aware, num_rounds, seed):
    label = f"GShield, evasion_lambda={lam}, calib_aware={calib_aware}"
    print(f"\n=== {label} ===", flush=True)
    sim = FederatedSimulation(
        clients, num_classes=num_classes, embed_dim=embed_dim,
        malicious_clients=malicious, attack="targeted",
        aggregation="gshield", seed=seed,
        scale_factor=1.0, evasion_lambda=lam, evasion_calib_aware=calib_aware,
    )
    t0 = time.time()
    logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    elapsed = time.time() - t0
    final = logs[-1]
    print(f"F1={final.macro_f1:.4f}, targeted_success={final.targeted_success_rate} ({elapsed:.1f}s)", flush=True)
    return {
        "defense": "gshield", "evasion_lambda": lam, "calib_aware": calib_aware,
        "final_macro_f1": final.macro_f1,
        "targeted_success_rate": final.targeted_success_rate,
        "wall_clock_seconds": elapsed,
    }


def run_rule_based(clients, num_classes, embed_dim, malicious, lam, calib_aware, num_rounds, seed):
    label = f"Rule-based (same-round z), evasion_lambda={lam}, calib_aware={calib_aware}"
    print(f"\n=== {label} ===", flush=True)
    policy = AdaptiveRuleBasedGatingPolicy(warmup_rounds=5, warmup_scale=1.5)
    sim = FederatedSimulation(
        clients, num_classes=num_classes, embed_dim=embed_dim,
        malicious_clients=malicious, attack="targeted",
        gating=True, seed=seed, gating_policy=policy,
        trust_decay=0.9, force_accept_round0=True,
        scale_factor=1.0, evasion_lambda=lam, evasion_calib_aware=calib_aware,
    )
    t0 = time.time()
    logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    elapsed = time.time() - t0
    final = logs[-1]
    malicious_excluded = sum(
        1 for h in sim.gating_history for c in malicious if h["actions"].get(c) == "exclude"
    )
    malicious_total = len(sim.gating_history) * len(malicious)
    print(f"F1={final.macro_f1:.4f}, targeted_success={final.targeted_success_rate}, "
          f"malicious_excluded={malicious_excluded}/{malicious_total} ({elapsed:.1f}s)", flush=True)
    return {
        "defense": "rule_based_adaptive_z", "evasion_lambda": lam, "calib_aware": calib_aware,
        "final_macro_f1": final.macro_f1,
        "targeted_success_rate": final.targeted_success_rate,
        "malicious_excluded_fraction": malicious_excluded / max(1, malicious_total),
        "wall_clock_seconds": elapsed,
    }


def main(n_clients=9, num_rounds=25, seed=0):
    print("Loading N-BaIoT ...", flush=True)
    clients = load_nbaiot_federated()
    classes = list(clients.values())[0].classes
    num_classes = len(classes)
    embed_dim = list(clients.values())[0].X.shape[1]
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    lambdas = [0.0, 0.9, 0.99]
    results = []

    for lam in lambdas:
        for calib_aware in [False, True]:
            if lam == 0.0 and calib_aware:
                continue  # calib-aware search needs a nonzero evasion floor to be meaningful
            results.append(run_gshield(clients, num_classes, embed_dim, malicious, lam, calib_aware, num_rounds, seed))
            with open(OUT_PATH, "w") as f:
                json.dump({"description": "Calibration-preserving adaptive attacker vs GShield and the rule-based z-score reference, targeted attack, N-BaIoT.", "runs": results}, f, indent=2)

    for lam in lambdas:
        for calib_aware in [False, True]:
            if lam == 0.0 and calib_aware:
                continue
            results.append(run_rule_based(clients, num_classes, embed_dim, malicious, lam, calib_aware, num_rounds, seed))
            with open(OUT_PATH, "w") as f:
                json.dump({"description": "Calibration-preserving adaptive attacker vs GShield and the rule-based z-score reference, targeted attack, N-BaIoT.", "runs": results}, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for r in results:
        print(f"  {r['defense']} lambda={r['evasion_lambda']} calib_aware={r['calib_aware']}: "
              f"F1={r['final_macro_f1']:.4f}, targeted_success={r['targeted_success_rate']}", flush=True)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
