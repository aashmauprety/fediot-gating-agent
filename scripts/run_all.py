"""Real experiment driver for Section VII-C (attack-type head validation),
VII-D (robustness under attack), and VII-E (trust-building attack) using
real N-BaIoT data. Writes CSV/JSON results to code/results/.

This uses the RuleBasedGatingPolicy stand-in wherever the paper's design
calls for the LLM Gating Agent -- see fedgate/gating_agent.py docstring.
Label results accordingly when copying them into the paper.

Usage:
    python3 run_all.py --stage backbone
    python3 run_all.py --stage robustness
    python3 run_all.py --stage trustbuild
    python3 run_all.py --stage all
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def stage_backbone(clients, num_classes, embed_dim, num_rounds):
    """VII-C: attack-type head validation, no attack, no gating."""
    print("=== VII-C: Attack-type head backbone (no attack) ===")
    sim = FederatedSimulation(
        clients, num_classes=num_classes, embed_dim=embed_dim,
        malicious_clients=[], attack="none", gating=False, seed=0,
    )
    t0 = time.time()
    logs = sim.run(num_rounds=num_rounds, log_every=1)
    elapsed = time.time() - t0
    out = {
        "description": "N-BaIoT attack-type head, federated (9 real devices as clients), no attack, no gating.",
        "num_rounds": num_rounds,
        "num_classes": num_classes,
        "classes": list(sim.splits[sim.client_ids[0]][0].classes),
        "wall_clock_seconds": elapsed,
        "rounds": sim.logs_as_dicts(),
        "final_macro_f1": logs[-1].macro_f1,
    }
    path = os.path.join(RESULTS_DIR, "backbone_nbaiot_attack_head.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"final macro-F1: {logs[-1].macro_f1:.4f}  (elapsed {elapsed:.1f}s) -> {path}")
    return out


def stage_robustness(clients, num_classes, embed_dim, num_rounds):
    """VII-D: robustness under untargeted and targeted attacks at several
    alpha, comparing plain FedAvg / trimmed-mean / Krum / rule-based
    Gating Agent."""
    print("=== VII-D: Robustness under attack ===")
    client_ids = list(clients.keys())
    n = len(client_ids)
    alphas = [0, 1, 2, 3, 4]  # out of 9 clients -> 0%, 11%, 22%, 33%, 44%
    strategies = [
        ("fedavg", False),
        ("trimmed_mean", False),
        ("krum", False),
        ("fedavg", True),  # gating=True uses the Gating Agent's own weighting
    ]
    results = []
    for attack_type in ("untargeted", "targeted"):
        for alpha_n in alphas:
            malicious = client_ids[:alpha_n]
            for agg, gating in strategies:
                label = "gating_agent" if gating else agg
                sim = FederatedSimulation(
                    clients, num_classes=num_classes, embed_dim=embed_dim,
                    malicious_clients=malicious, attack=attack_type if alpha_n > 0 else "none",
                    gating=gating, aggregation=agg, seed=0,
                    targeted_source=1, targeted_target=0,
                )
                t0 = time.time()
                logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)  # only need final
                elapsed = time.time() - t0
                final = logs[-1]
                row = {
                    "attack_type": attack_type,
                    "alpha_clients": alpha_n,
                    "alpha_frac": alpha_n / n,
                    "strategy": label,
                    "final_macro_f1": final.macro_f1,
                    "targeted_success_rate": final.targeted_success_rate,
                    "wall_clock_seconds": elapsed,
                }
                results.append(row)
                print(
                    f"  attack={attack_type:10s} alpha={alpha_n}/{n} strategy={label:14s} "
                    f"F1={final.macro_f1:.3f} "
                    f"targeted_success={final.targeted_success_rate}"
                )
    path = os.path.join(RESULTS_DIR, "robustness_nbaiot.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {path}")
    return results


def stage_trustbuild(clients, num_classes, embed_dim, num_rounds):
    """VII-E: trust-building (delayed) attack vs the Bayesian-informed
    rule-based Gating Agent, at fixed alpha, sweeping delay D."""
    print("=== VII-E: Trust-building attack ===")
    client_ids = list(clients.keys())
    malicious = client_ids[:3]  # alpha ~ 33%
    delays = [0, 10, 20]
    results = []
    for delay in delays:
        for gating in (False, True):
            label = "gating_agent" if gating else "fedavg"
            sim = FederatedSimulation(
                clients, num_classes=num_classes, embed_dim=embed_dim,
                malicious_clients=malicious, attack="trust_building",
                gating=gating, trust_building_delay=delay, seed=0,
                targeted_source=1, targeted_target=0,
            )
            t0 = time.time()
            logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
            elapsed = time.time() - t0
            final = logs[-1]
            row = {
                "delay_rounds": delay,
                "strategy": label,
                "final_macro_f1": final.macro_f1,
                "targeted_success_rate": final.targeted_success_rate,
                "wall_clock_seconds": elapsed,
            }
            results.append(row)
            print(
                f"  delay={delay:3d} strategy={label:14s} F1={final.macro_f1:.3f} "
                f"targeted_success={final.targeted_success_rate}"
            )
    path = os.path.join(RESULTS_DIR, "trustbuild_nbaiot.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"-> {path}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["backbone", "robustness", "trustbuild", "all"], default="all")
    ap.add_argument("--n_max_per_class", type=int, default=3000)
    ap.add_argument("--num_rounds", type=int, default=25)
    args = ap.parse_args()

    print(f"Loading N-BaIoT (n_max_per_class={args.n_max_per_class}) ...")
    t0 = time.time()
    clients = load_nbaiot_federated(n_max_per_class=args.n_max_per_class, fine_grained=False, seed=0)
    print(f"Loaded {len(clients)} clients in {time.time()-t0:.1f}s")

    num_classes = 3
    embed_dim = 115

    if args.stage in ("backbone", "all"):
        stage_backbone(clients, num_classes, embed_dim, args.num_rounds)
    if args.stage in ("robustness", "all"):
        stage_robustness(clients, num_classes, embed_dim, args.num_rounds)
    if args.stage in ("trustbuild", "all"):
        stage_trustbuild(clients, num_classes, embed_dim, args.num_rounds)

    print("\nAll requested stages complete.")
