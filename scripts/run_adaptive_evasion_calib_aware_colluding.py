"""Colluding calibration-preserving adaptive attacker (follow-up to the
isolated calibration-aware attacker, README "thirty-sixth real result"):
that attacker's blind spot was structural, not a bug -- it minimized its
own ABSOLUTE calibration impact (measured on its own val split) because
it cannot see other clients' calib_loss_delta this round, so it cannot
directly target the real policy's same-round cross-client z-score.

A colluding attacker (malicious clients sharing local data statistics
with each other, the same stronger threat-model assumption already used
for `evasion_collude`'s direction-blending reference) gets a LESS NOISY
estimate of its own impact by pooling all malicious clients' val splits
(`fedgate/experiment.py:_local_calib_proxy_loss_pooled`), and -- since
`round_calib_deltas` for the z-score is computed over EVERY client this
round, malicious included -- a correlated, jointly-chosen lambda could
in principle shift the population statistics themselves, not just hide
from them. Still cannot see honest clients' deltas directly; this
tests whether pooling malicious-only information is enough.

`FederatedSimulation(evasion_collude=True, evasion_calib_aware=True)`
already composes both mechanisms (the existing `evasion_collude` flag
controls both the direction reference AND, now, which calibration
proxy `_select_calib_aware_lambda` judges against). Same protocol as
the isolated-attacker test: N-BaIoT, targeted attack, GShield (control)
and `AdaptiveRuleBasedGatingPolicy` (real z-score reference).
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
OUT_PATH = os.path.join(RESULTS_DIR, "adaptive_evasion_calib_aware_colluding.json")


def run_gshield(clients, num_classes, embed_dim, malicious, lam, num_rounds, seed):
    label = f"GShield, colluding calib-aware, evasion_lambda={lam}"
    print(f"\n=== {label} ===", flush=True)
    sim = FederatedSimulation(
        clients, num_classes=num_classes, embed_dim=embed_dim,
        malicious_clients=malicious, attack="targeted",
        aggregation="gshield", seed=seed,
        scale_factor=1.0, evasion_lambda=lam, evasion_collude=True, evasion_calib_aware=True,
    )
    t0 = time.time()
    logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    elapsed = time.time() - t0
    final = logs[-1]
    print(f"F1={final.macro_f1:.4f}, targeted_success={final.targeted_success_rate} ({elapsed:.1f}s)", flush=True)
    return {
        "defense": "gshield", "evasion_lambda": lam,
        "final_macro_f1": final.macro_f1,
        "targeted_success_rate": final.targeted_success_rate,
        "wall_clock_seconds": elapsed,
    }


def run_rule_based(clients, num_classes, embed_dim, malicious, lam, num_rounds, seed):
    label = f"Rule-based (same-round z), colluding calib-aware, evasion_lambda={lam}"
    print(f"\n=== {label} ===", flush=True)
    policy = AdaptiveRuleBasedGatingPolicy(warmup_rounds=5, warmup_scale=1.5)
    sim = FederatedSimulation(
        clients, num_classes=num_classes, embed_dim=embed_dim,
        malicious_clients=malicious, attack="targeted",
        gating=True, seed=seed, gating_policy=policy,
        trust_decay=0.9, force_accept_round0=True,
        scale_factor=1.0, evasion_lambda=lam, evasion_collude=True, evasion_calib_aware=True,
    )
    t0 = time.time()
    logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    elapsed = time.time() - t0
    final = logs[-1]
    malicious_excluded = sum(
        1 for h in sim.gating_history for c in malicious if h["actions"].get(c) == "exclude"
    )
    malicious_total = len(sim.gating_history) * len(malicious)
    # also pull the max same-round calib z-score the malicious clients hit,
    # to check whether pooling actually reduced their outlier-ness at all
    import re
    max_calib_z = 0.0
    for h in sim.gating_history:
        for c in malicious:
            rationale = h["rationales"].get(c, "")
            m = re.search(r"calib-loss same-round z-score=(-?[\d.]+)", rationale)
            if m:
                max_calib_z = max(max_calib_z, abs(float(m.group(1))))
    print(f"F1={final.macro_f1:.4f}, targeted_success={final.targeted_success_rate}, "
          f"malicious_excluded={malicious_excluded}/{malicious_total}, max|calib_z|={max_calib_z:.1f} ({elapsed:.1f}s)", flush=True)
    return {
        "defense": "rule_based_adaptive_z", "evasion_lambda": lam,
        "final_macro_f1": final.macro_f1,
        "targeted_success_rate": final.targeted_success_rate,
        "malicious_excluded_fraction": malicious_excluded / max(1, malicious_total),
        "max_abs_calib_z": max_calib_z,
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

    lambdas = [0.9, 0.99]
    results = []

    for lam in lambdas:
        results.append(run_gshield(clients, num_classes, embed_dim, malicious, lam, num_rounds, seed))
        with open(OUT_PATH, "w") as f:
            json.dump({"description": "Colluding calibration-aware adaptive attacker vs GShield and the rule-based z-score reference, targeted attack, N-BaIoT.", "runs": results}, f, indent=2)

    for lam in lambdas:
        results.append(run_rule_based(clients, num_classes, embed_dim, malicious, lam, num_rounds, seed))
        with open(OUT_PATH, "w") as f:
            json.dump({"description": "Colluding calibration-aware adaptive attacker vs GShield and the rule-based z-score reference, targeted attack, N-BaIoT.", "runs": results}, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for r in results:
        print(f"  {r['defense']} lambda={r['evasion_lambda']}: F1={r['final_macro_f1']:.4f}, "
              f"targeted_success={r['targeted_success_rate']}", flush=True)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
