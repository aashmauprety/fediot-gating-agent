"""Adaptive-attacker experiment (follow-up to the GShield-style baseline,
README "sixteenth real result" / main.tex Table XII): an attacker who
KNOWS the defense flags updates whose direction deviates from the
honest consensus can blend its poisoned update's direction toward a
plausible reference direction (the previous round's aggregated global
update -- any client can compute this from two consecutive broadcast
global models) to evade detection, at the cost of diluting the attack's
effect. This is a standard "constrain-and-scale"-style adaptive attack
from the poisoning literature (cf. bagdasaryan2020backdoor, already
cited in the paper's threat model), not a novel attack design -- what's
new here is testing whether it defeats the GShield-style clustering
baseline more easily than the LLM Gating Agent.

`FederatedSimulation(evasion_lambda=...)` blends direction (see
fedgate/attacks.py:blend_toward_reference); we also drop scale_factor to
1.0 (no magnitude amplification) since blending is a DIRECTION-only
evasion and an adaptive attacker sophisticated enough to blend direction
would not naively also amplify magnitude 5x, which is trivially caught
by any magnitude-sensitive defense regardless of direction.

Phase 1 (this script): sweep evasion_lambda against the GShield-style
baseline only (cheap, seconds per run) across both attack types on CIC
IoT-DIAD 2024, to find where/whether it degrades. Phase 2 (a follow-up
script) will re-run the most informative lambda value(s) against the
qwen2.5:14b hybrid Gating Agent for a direct comparison.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_ciciot_diad_attack_classification
from fedgate.experiment import FederatedSimulation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "adaptive_evasion_gshield_sweep.json")


def main(n_max_per_class=5000, n_clients=9, num_rounds=25, seed=0):
    print("Loading CIC IoT-DIAD 2024 attack data ...", flush=True)
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    lambdas = [0.0, 0.5, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]
    results = []
    for attack in ["untargeted", "targeted"]:
        for lam in lambdas:
            label = f"{attack}, evasion_lambda={lam}"
            print(f"\n=== {label} ===", flush=True)
            sim = FederatedSimulation(
                clients, num_classes=len(classes), embed_dim=embed_dim,
                malicious_clients=malicious, attack=attack,
                aggregation="gshield", seed=seed,
                scale_factor=1.0, evasion_lambda=lam,
            )
            t0 = time.time()
            logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
            elapsed = time.time() - t0
            final = logs[-1]
            print(f"F1={final.macro_f1:.4f}, targeted_success={final.targeted_success_rate} ({elapsed:.1f}s)", flush=True)
            results.append({
                "attack": attack, "evasion_lambda": lam,
                "final_macro_f1": final.macro_f1,
                "targeted_success_rate": final.targeted_success_rate,
                "wall_clock_seconds": elapsed,
            })
            with open(OUT_PATH, "w") as f:
                json.dump({
                    "description": (
                        "GShield-style baseline under a direction-blending adaptive "
                        "attacker (scale_factor=1.0, no magnitude amplification), "
                        "sweeping evasion_lambda in [0,1] on CIC IoT-DIAD 2024."
                    ),
                    "runs": results,
                }, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for r in results:
        print(f"  {r['attack']} lambda={r['evasion_lambda']}: F1={r['final_macro_f1']:.4f}, "
              f"targeted_success={r['targeted_success_rate']}", flush=True)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
