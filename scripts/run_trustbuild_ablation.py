"""Real round-only Bayesian-prior ablation (Section VII-E TODO item 1):
isolates the Bayesian long-term trust posterior's specific contribution
from the rest of the Gating Agent, on the same real N-BaIoT trust-building
setup as run_all.py's stage_trustbuild.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(n_max_per_class=3000, num_rounds=25):
    print(f"Loading N-BaIoT (n_max_per_class={n_max_per_class}) ...")
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]  # alpha ~ 33%, same as run_all.py's stage_trustbuild
    delays = [0, 10, 20]
    results = []
    for delay in delays:
        for use_bayesian in (True, False):
            label = "gating_with_bayesian" if use_bayesian else "gating_round_only"
            sim = FederatedSimulation(
                clients, num_classes=3, embed_dim=115,
                malicious_clients=malicious, attack="trust_building",
                gating=True, trust_building_delay=delay, seed=0,
                use_bayesian_prior=use_bayesian,
            )
            t0 = time.time()
            logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
            elapsed = time.time() - t0
            final = logs[-1]
            row = {
                "delay_rounds": delay,
                "strategy": label,
                "final_macro_f1": final.macro_f1,
                "wall_clock_seconds": elapsed,
            }
            results.append(row)
            print(f"  delay={delay:3d} strategy={label:22s} F1={final.macro_f1:.3f}")
    path = os.path.join(RESULTS_DIR, "trustbuild_ablation_nbaiot.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
