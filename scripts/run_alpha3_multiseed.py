"""Multi-seed re-run of the untargeted alpha=3/9 Gating Agent anomaly
flagged as a TODO in Section VII-D (Table tab:nbaiot_robustness): a single
seed showed the rule-based Gating Agent underperforming even trimmed-mean
at this one alpha, attributed to a small-sample (9-client) update-norm
z-score. This checks whether that's a one-off across seeds 0-4.
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
    seeds = [0, 1, 2, 3, 4]
    results = []
    for seed in seeds:
        clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=seed)
        client_ids = list(clients.keys())
        malicious = client_ids[:3]
        sim = FederatedSimulation(
            clients, num_classes=3, embed_dim=115,
            malicious_clients=malicious, attack="untargeted",
            gating=True, seed=seed,
        )
        t0 = time.time()
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
        elapsed = time.time() - t0
        f1 = logs[-1].macro_f1
        results.append({"seed": seed, "final_macro_f1": f1, "wall_clock_seconds": elapsed})
        print(f"  seed={seed} F1={f1:.3f} ({elapsed:.1f}s)")
    path = os.path.join(RESULTS_DIR, "alpha3_gating_multiseed.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
