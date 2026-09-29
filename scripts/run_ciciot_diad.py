"""Real first test of CIC IoT-DIAD 2024 (code/data/labeled_data/), a
dataset already staged locally with real per-device labels AND real
attack labels in one place, using a much richer per-packet/flow feature
set (inter-arrival time, TTL, TCP window size, payload entropy, TLS/
HTTP/DNS/ICMP fields, multi-window traffic statistics) than anything
else used in this paper so far -- see fedgate/data.py's module docstring
for the full story and what was already excluded (identity/leakage
columns, high-cardinality text fields).

Two real federated experiments:
  (A) Device-type identification: genuine per-real-device federated
      clients (like N-BaIoT's 9 devices, just with up to 59 here).
  (B) Attack-type classification: same richer features, pooled-row
      client partition (no device labels in this file, same real
      limitation as CIC IoT 2023).

Both use robust (median/IQR) feature scaling, the same approach used for
N-BaIoT/CIC IoT 2023's heavy-tailed flow-statistic features elsewhere in
this codebase, computed here explicitly since classifier_training.py's
federated_train_classifier does not scale internally.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import numpy as np

from fedgate.data import load_ciciot_diad_device_identification, load_ciciot_diad_attack_classification
from fedgate.classifier_training import federated_train_classifier

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def robust_scale(client_X: dict) -> dict:
    all_X = np.concatenate(list(client_X.values()), axis=0).astype(np.float64)
    median = np.median(all_X, axis=0)
    q75, q25 = np.percentile(all_X, [75, 25], axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-6] = 1.0
    return {
        cid: np.clip((X.astype(np.float64) - median) / iqr, -20.0, 20.0).astype(np.float32)
        for cid, X in client_X.items()
    }


def stage_device_identification(n_max_per_device=2000, num_rounds=25, seed=0, n_clients=10):
    print("=== (A) CIC IoT-DIAD 2024 device identification ===")
    print(f"NOTE: n_clients={n_clients} buckets multiple real devices per federated "
          f"client (like UNSW), NOT one client per device -- see module comment in "
          f"data.py and code/README.md for why one-client-per-device is fatal for a "
          f"device-IDENTIFICATION task specifically (each client's local label would "
          f"be constant, degenerating FedAvg to averaging single-class specialists).")
    t0 = time.time()
    clients = load_ciciot_diad_device_identification(
        n_max_per_device=n_max_per_device, min_rows_per_device=50, seed=seed,
        n_clients=n_clients,
    )
    print(f"Loaded {len(clients)} real devices in {time.time()-t0:.1f}s")
    classes = list(clients.values())[0]["classes"]
    total_rows = sum(len(c["y_device"]) for c in clients.values())
    print(f"{len(classes)} device classes, {total_rows} total rows")

    client_X = {cid: c["X"] for cid, c in clients.items()}
    client_X = robust_scale(client_X)
    client_y = {cid: c["y_device"] for cid, c in clients.items()}

    t0 = time.time()
    result = federated_train_classifier(
        client_X, client_y, num_classes=len(classes), embed_dim=client_X[list(client_X)[0]].shape[1],
        num_rounds=num_rounds, seed=seed,
    )
    elapsed = time.time() - t0
    print(f"Device-ID: final macro-F1 = {result['final_macro_f1']:.4f} ({elapsed:.1f}s, {num_rounds} rounds)")

    out = {
        "description": "Real CIC IoT-DIAD 2024 device identification, genuine per-device federated clients, "
                        "118 rich numeric features (IAT, TTL, TCP window, TLS/HTTP/DNS/ICMP fields, multi-window stats)",
        "n_devices": len(classes),
        "n_clients": len(clients),
        "total_rows": total_rows,
        "num_rounds": num_rounds,
        "final_macro_f1": result["final_macro_f1"],
        "wall_clock_seconds": elapsed,
        "rounds": result["rounds"],
    }
    path = os.path.join(RESULTS_DIR, "ciciot_diad_device_identification.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {path}")
    return out


def stage_attack_classification(n_max_per_class=5000, num_rounds=25, seed=0):
    print("\n=== (B) CIC IoT-DIAD 2024 attack classification, richer features ===")
    t0 = time.time()
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=9, label_col="attack_group", seed=seed,
    )
    print(f"Loaded {len(clients)} clients in {time.time()-t0:.1f}s")
    classes = list(clients.values())[0].classes
    total_rows = sum(len(c.y_attack) for c in clients.values())
    print(f"{len(classes)} attack classes: {classes}, {total_rows} total rows")

    client_X = {cid: c.X for cid, c in clients.items()}
    client_X = robust_scale(client_X)
    client_y = {cid: c.y_attack for cid, c in clients.items()}

    t0 = time.time()
    result = federated_train_classifier(
        client_X, client_y, num_classes=len(classes), embed_dim=client_X[list(client_X)[0]].shape[1],
        num_rounds=num_rounds, seed=seed,
    )
    elapsed = time.time() - t0
    print(f"Attack classification: final macro-F1 = {result['final_macro_f1']:.4f} ({elapsed:.1f}s, {num_rounds} rounds)")

    out = {
        "description": "Real CIC IoT-DIAD 2024 attack-group classification, pooled-row client partition "
                        "(no device labels in this file), 118 rich numeric features",
        "classes": classes,
        "n_clients": len(clients),
        "total_rows": total_rows,
        "num_rounds": num_rounds,
        "final_macro_f1": result["final_macro_f1"],
        "wall_clock_seconds": elapsed,
        "rounds": result["rounds"],
    }
    path = os.path.join(RESULTS_DIR, "ciciot_diad_attack_classification.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"-> {path}")
    return out


if __name__ == "__main__":
    stage_device_identification()
    stage_attack_classification()
    print("\nAll CIC IoT-DIAD 2024 stages complete.")
