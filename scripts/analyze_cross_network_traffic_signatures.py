"""Pure data analysis, no training: investigates the real, striking
pattern from Section VII-G's five-technique investigation -- Amazon Echo
is the only device that ever cross-network-matches correctly, regardless
of training objective (self-supervised, domain-adversarial x3, or
explicit contrastive alignment). This computes simple, interpretable
statistics directly on the raw signed packet-size windows (the SAME
representation the encoder actually consumes -- no timestamps, no other
side info) for all 4 aligned devices, in both UNSW and YourThings
captures, to check a concrete hypothesis: does Amazon Echo's raw
packet-size signature shift LESS between the two networks than the other
three devices' do?
"""
import json
import os
import sys

sys.path.insert(0, "..")

import numpy as np

from fedgate.data import load_unsw_device_identification, load_yourthings_device_identification

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")

YOURTHINGS_TO_UNSW_ALIGNMENT = {
    "SamsungSmartThingsHub": "Smart Things",
    "AmazonEchoGen1": "Amazon Echo",
    "BelkinWeMoMotionSensor": "Belkin wemo motion sensor",
    "LIFXVirtualBulb": "Light Bulbs LiFX Smart Bulb",
}


def packet_size_stats(X: np.ndarray) -> dict:
    """X: (n_windows, seq_len) signed packet sizes. Real, simple
    descriptive statistics of the raw representation the encoder
    consumes -- nothing derived from timestamps or any info the model
    doesn't see."""
    flat = X.flatten()
    abs_sizes = np.abs(flat)
    # mode concentration: what fraction of packets share the single most
    # common exact size (a proxy for "fixed-size protocol/heartbeat
    # packets" vs. "variable-size data traffic")
    vals, counts = np.unique(abs_sizes, return_counts=True)
    top_frac = counts.max() / len(abs_sizes)
    top_size = vals[np.argmax(counts)]
    # direction balance: fraction of packets that are incoming (positive)
    frac_incoming = float((flat > 0).mean())
    return {
        "mean_abs_size": float(abs_sizes.mean()),
        "std_abs_size": float(abs_sizes.std()),
        "median_abs_size": float(np.median(abs_sizes)),
        "top_size_value": float(top_size),
        "top_size_fraction": float(top_frac),
        "frac_incoming": frac_incoming,
        "n_packets": int(len(flat)),
    }


def cross_network_shift(unsw_stats: dict, yt_stats: dict) -> dict:
    """How much does each statistic differ between the two networks'
    captures of the (putatively) same device type? Normalized by the
    UNSW-side std where relevant, so devices with naturally noisier
    traffic aren't unfairly penalized for a given absolute shift."""
    pooled_std = max(unsw_stats["std_abs_size"], 1e-6)
    mean_shift_z = abs(unsw_stats["mean_abs_size"] - yt_stats["mean_abs_size"]) / pooled_std
    top_size_match = unsw_stats["top_size_value"] == yt_stats["top_size_value"]
    top_frac_diff = abs(unsw_stats["top_size_fraction"] - yt_stats["top_size_fraction"])
    direction_diff = abs(unsw_stats["frac_incoming"] - yt_stats["frac_incoming"])
    return {
        "mean_size_shift_z": mean_shift_z,
        "top_size_matches_exactly": top_size_match,
        "top_size_fraction_diff": top_frac_diff,
        "direction_balance_diff": direction_diff,
    }


def main():
    print("Loading real UNSW data ...")
    unsw_clients = load_unsw_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=100,
        n_clients=10, seed=0,
    )
    unsw_classes = unsw_clients["unsw_client_0"]["classes"]
    unsw_by_class = {c: [] for c in unsw_classes}
    for cid, c in unsw_clients.items():
        for i, lbl in enumerate(c["y_device"]):
            unsw_by_class[unsw_classes[lbl]].append(c["X"][i])
    unsw_by_class = {c: np.stack(v) for c, v in unsw_by_class.items() if v}

    print("Loading real YourThings data (IP-verified) ...")
    yt_clients = load_yourthings_device_identification(
        seq_len=100, max_windows_per_device=2000, min_windows_per_device=5,
        n_clients=1, seed=0,
    )
    yt_classes = yt_clients["yourthings_client_0"]["classes"]
    yt_X_all = yt_clients["yourthings_client_0"]["X"]
    yt_y_all = yt_clients["yourthings_client_0"]["y_device"]

    rows = []
    print(f"\n{'Device':30s} {'MeanShift(z)':>13s} {'TopSizeMatch':>13s} {'TopFracDiff':>12s} {'DirDiff':>8s}")
    for yt_name, unsw_cls in YOURTHINGS_TO_UNSW_ALIGNMENT.items():
        if yt_name not in yt_classes:
            continue
        idx = yt_classes.index(yt_name)
        yt_X = yt_X_all[yt_y_all == idx]
        unsw_X = unsw_by_class[unsw_cls]

        u_stats = packet_size_stats(unsw_X)
        y_stats = packet_size_stats(yt_X)
        shift = cross_network_shift(u_stats, y_stats)

        rows.append({
            "yourthings_device": yt_name,
            "unsw_class": unsw_cls,
            "unsw_stats": u_stats,
            "yourthings_stats": y_stats,
            "cross_network_shift": shift,
            "cross_network_retrieval_correct_in_5_technique_investigation":
                yt_name == "AmazonEchoGen1",
        })
        print(f"{yt_name:30s} {shift['mean_size_shift_z']:13.2f} "
              f"{str(shift['top_size_matches_exactly']):>13s} "
              f"{shift['top_size_fraction_diff']:12.3f} {shift['direction_balance_diff']:8.3f}")

    # Rank devices by mean_size_shift_z -- lower shift is the concrete
    # hypothesis for "more network-invariant, easier to cross-network-match"
    ranked = sorted(rows, key=lambda r: r["cross_network_shift"]["mean_size_shift_z"])
    print("\nRanked by cross-network mean-size shift (lowest = most stable across networks):")
    for r in ranked:
        marker = " <-- the one that succeeds in retrieval" if r["yourthings_device"] == "AmazonEchoGen1" else ""
        print(f"  {r['yourthings_device']:30s} shift_z={r['cross_network_shift']['mean_size_shift_z']:.2f}{marker}")

    out = {"description": "Cross-network raw packet-size signature analysis for the 4 aligned devices",
           "rows": rows, "ranked_by_shift": [r["yourthings_device"] for r in ranked]}
    path = os.path.join(RESULTS_DIR, "cross_network_traffic_signature_analysis.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
