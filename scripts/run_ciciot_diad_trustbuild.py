"""Real rule-based baseline for the Gating Agent on CIC IoT-DIAD 2024's
much richer feature set (118 numeric features -- IAT, TTL, TCP window,
TLS/HTTP/DNS/ICMP fields, multi-window traffic stats -- vs. N-BaIoT's
115 flow-statistic features). This establishes the reference point
before testing whether richer per-round digests change how an LLM-based
Gating Agent behaves (the current focus per the user: fix the LLM
Gating Agent weakness, using this richer dataset as the testbed, not
pursuing cross-network generalization for now).

Same protocol as run_trustbuild_ablation.py (N-BaIoT): trust-building
attack, delays in {0, 10, 20}, rule-based Gating Agent with vs. without
the Bayesian prior (round-only ablation), 25 rounds. `load_ciciot_diad_attack_classification`
already returns ClientData objects, so this is a direct drop-in swap of
the data source into the same FederatedSimulation used throughout
Section VII -- no new plumbing needed.

Real caveat carried over from CIC IoT 2023 (documented in
fedgate/data.py): attack_task.parquet has no device labels, so these are
pooled-row clients, not genuine per-device federated clients -- same
limitation as CIC IoT 2023, not specific to this experiment.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_ciciot_diad_attack_classification
from fedgate.experiment import FederatedSimulation

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def main(n_max_per_class=5000, n_clients=9, num_rounds=25):
    print(f"Loading CIC IoT-DIAD 2024 attack data (n_max_per_class={n_max_per_class}, "
          f"n_clients={n_clients}) ...")
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    print(f"{len(classes)} classes: {classes}, embed_dim={embed_dim}, {len(clients)} clients")

    client_ids = list(clients.keys())
    malicious = client_ids[:3]  # alpha ~ 3/9, same ratio as N-BaIoT trust-building setup
    delays = [0, 10, 20]
    results = []
    for delay in delays:
        for use_bayesian in (True, False):
            label = "gating_with_bayesian" if use_bayesian else "gating_round_only"
            sim = FederatedSimulation(
                clients, num_classes=len(classes), embed_dim=embed_dim,
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
            print(f"  delay={delay:3d} strategy={label:22s} F1={final.macro_f1:.3f} ({elapsed:.1f}s)")

    # Also plain FedAvg (no gating) at each delay, for the same reference
    # comparison N-BaIoT's table uses.
    for delay in delays:
        sim = FederatedSimulation(
            clients, num_classes=len(classes), embed_dim=embed_dim,
            malicious_clients=malicious, attack="trust_building",
            gating=False, trust_building_delay=delay, seed=0,
        )
        t0 = time.time()
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
        elapsed = time.time() - t0
        final = logs[-1]
        row = {"delay_rounds": delay, "strategy": "fedavg_plain",
               "final_macro_f1": final.macro_f1, "wall_clock_seconds": elapsed}
        results.append(row)
        print(f"  delay={delay:3d} strategy={'fedavg_plain':22s} F1={final.macro_f1:.3f} ({elapsed:.1f}s)")

    out = {
        "description": "CIC IoT-DIAD 2024 trust-building attack, rule-based Gating Agent baseline "
                        "(richer 118-feature digest basis vs. N-BaIoT's 115)",
        "classes": classes,
        "embed_dim": embed_dim,
        "n_clients": len(clients),
        "results": results,
    }
    path = os.path.join(RESULTS_DIR, "ciciot_diad_trustbuild_rulebased.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
