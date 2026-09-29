"""GShield-style non-LLM robust-aggregation baseline (README "still open
item 1", now being done -- deferred to last per user instruction, this is
last). Inspired by GShield (Sameera et al., arXiv 2512.19286, Dec 2025):
PCA-reduce each round's flattened client updates, 2-means cluster them,
fit a Gaussian to the larger (benign) cluster, and exclude any client
whose Mahalanobis distance to that Gaussian is an outlier
(`fedgate.federated.gshield_aggregate`) -- no LLM calls, no assumption
about the number of malicious clients (unlike Krum).

Runs the identical set of configurations already used to validate the
Qwen2.5 14B hybrid Gating Agent, on both datasets, for a direct,
current (2025-era) non-LLM comparison point:
  - CIC IoT-DIAD 2024: trust_building D=10 (seeds 0,1,2), D=20,
    untargeted, targeted.
  - N-BaIoT: trust_building D=0/10/20, untargeted, targeted.

Reference numbers already on record (see code/README.md and main.tex
Tables IX-XI):
  CIC IoT-DIAD, rule-based:            D=10: 0.758
  CIC IoT-DIAD, Qwen2.5 14B hybrid:    0.895-0.912 across all 6 configs
  N-BaIoT, rule-based (Bayesian):      D=0: 0.320  D=10: 0.986  D=20: 0.988
  N-BaIoT, Qwen2.5 14B hybrid:         0.9958-0.9974 across all 5 configs
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_ciciot_diad_attack_classification, load_nbaiot_federated
from fedgate.experiment import FederatedSimulation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "gshield_baseline.json")


def run_one(clients, num_classes, embed_dim, malicious, attack, delay, num_rounds, seed, label):
    print(f"\n=== {label} ===", flush=True)
    kwargs = dict(
        clients=clients, num_classes=num_classes, embed_dim=embed_dim,
        malicious_clients=malicious, attack=attack,
        aggregation="gshield", seed=seed,
    )
    if attack == "trust_building":
        kwargs["trust_building_delay"] = delay
    sim = FederatedSimulation(**kwargs)
    t0 = time.time()
    logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    elapsed = time.time() - t0
    final = logs[-1]
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s)", flush=True)
    return {
        "label": label, "attack": attack, "delay": delay, "seed": seed,
        "final_macro_f1": final.macro_f1,
        "targeted_success_rate": final.targeted_success_rate,
        "wall_clock_seconds": elapsed,
    }


def main():
    results = {"ciciot_diad": [], "nbaiot": []}

    print("=== CIC IoT-DIAD 2024 ===", flush=True)
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=5000, n_clients=9, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    ciciot_runs = [
        ("trust_building", 10, 0, "trust_building D=10 seed=0"),
        ("trust_building", 10, 1, "trust_building D=10 seed=1"),
        ("trust_building", 10, 2, "trust_building D=10 seed=2"),
        ("trust_building", 20, 0, "trust_building D=20 seed=0"),
        ("untargeted", 0, 0, "untargeted seed=0"),
        ("targeted", 0, 0, "targeted seed=0"),
    ]
    for attack, delay, seed, label in ciciot_runs:
        out = run_one(clients, len(classes), embed_dim, malicious, attack, delay, 25, seed, label)
        results["ciciot_diad"].append(out)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== N-BaIoT ===", flush=True)
    nb_clients = load_nbaiot_federated(n_max_per_class=3000, fine_grained=False, seed=0)
    nb_client_ids = list(nb_clients.keys())
    nb_malicious = nb_client_ids[:3]

    nbaiot_runs = [
        ("trust_building", 0, 0, "trust_building D=0"),
        ("trust_building", 10, 0, "trust_building D=10"),
        ("trust_building", 20, 0, "trust_building D=20"),
        ("untargeted", 0, 0, "untargeted"),
        ("targeted", 0, 0, "targeted"),
    ]
    for attack, delay, seed, label in nbaiot_runs:
        out = run_one(nb_clients, 3, 115, nb_malicious, attack, delay, 25, seed, label)
        results["nbaiot"].append(out)
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Summary ===", flush=True)
    print("CIC IoT-DIAD (rule-based D=10: 0.758, Qwen14B hybrid: 0.895-0.912):")
    for r in results["ciciot_diad"]:
        print(f"  {r['label']}: F1={r['final_macro_f1']:.4f}")
    print("N-BaIoT (rule-based D=0/10/20: 0.320/0.986/0.988, Qwen14B hybrid: 0.9958-0.9974):")
    for r in results["nbaiot"]:
        print(f"  {r['label']}: F1={r['final_macro_f1']:.4f}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
