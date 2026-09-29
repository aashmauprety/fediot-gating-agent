"""Real experiment on real CIC IoT 2023 data (extends Section VII-C):
trains the attack-type head federatedly on the full 8-class taxonomy
(Benign, BruteForce, DDoS, DoS, Mirai, Recon, Spoofing, Web-based),
across a random 9-client partition (no device-level identifier available
in this merged/flattened format -- see fedgate/data.py docstring).
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_ciciot2023_federated, CICIOT2023_CLASSES
from fedgate.experiment import FederatedSimulation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def main(n_per_class=4000, n_clients=9, num_rounds=25):
    print(f"Loading real CIC IoT 2023 data (n_per_class={n_per_class}) ...")
    t0 = time.time()
    clients = load_ciciot2023_federated(n_per_class=n_per_class, n_clients=n_clients, split="train", seed=0)
    print(f"Loaded {len(clients)} clients in {time.time()-t0:.1f}s")
    total = sum(len(c.y_attack) for c in clients.values())
    print(f"Total samples: {total}, classes: {CICIOT2023_CLASSES}")

    embed_dim = clients["ciciot_client_0"].X.shape[1]
    print(f"Feature dim: {embed_dim}")

    sim = FederatedSimulation(
        clients, num_classes=len(CICIOT2023_CLASSES), embed_dim=embed_dim,
        malicious_clients=[], attack="none", gating=False, seed=0,
    )
    t0 = time.time()
    logs = sim.run(num_rounds=num_rounds, log_every=1)
    elapsed = time.time() - t0
    print(f"Final macro-F1: {logs[-1].macro_f1:.4f} (elapsed {elapsed:.1f}s)")

    out = {
        "description": "Real CIC IoT 2023 attack-type head, federated (9 random-partition clients, no device identifier available), no attack, no gating",
        "num_rounds": num_rounds,
        "num_classes": len(CICIOT2023_CLASSES),
        "classes": CICIOT2023_CLASSES,
        "n_per_class": n_per_class,
        "total_samples": total,
        "wall_clock_seconds": elapsed,
        "rounds": sim.logs_as_dicts(),
        "final_macro_f1": logs[-1].macro_f1,
    }
    path = os.path.join(RESULTS_DIR, "ciciot2023_attack_head.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {path}")


if __name__ == "__main__":
    main()
