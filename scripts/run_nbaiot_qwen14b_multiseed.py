"""Close a documented coverage gap (README 'still open' item): N-BaIoT's
trust_building D=10 headline number (Table III of the ICC paper) has
only ever used seed=0, unlike CIC IoT-DIAD's same config which used 3
seeds. Reference: N-BaIoT full-LLM hybrid, D=10, seed=0 -> F1=0.9974.
This script adds seeds 1 and 2, identical config otherwise.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import LLMExternalBayesianPolicy, LLMGatingPolicyError

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "nbaiot_qwen14b_multiseed.json")
MODEL = "qwen2.5:14b"


def run_one(clients, malicious, seed, num_rounds):
    label = f"N-BaIoT, qwen2.5:14b hybrid, TB D=10, seed={seed}"
    print(f"\n=== {label} ===", flush=True)
    policy = LLMExternalBayesianPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
        timeout_s=60.0, max_retries=3,
    )
    sim = FederatedSimulation(
        clients, num_classes=3, embed_dim=115,
        malicious_clients=malicious, attack="trust_building",
        gating=True, trust_building_delay=10, seed=seed,
        gating_policy=policy, trust_decay=0.9, force_accept_round0=True,
    )
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        print(f"FAILED: {e}", flush=True)
        return {"label": label, "seed": seed, "error": str(e)}
    elapsed = time.time() - t0
    final = logs[-1]
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
          f"fallback {n_fallback_rounds}/{len(sim.gating_history)}", flush=True)
    return {
        "label": label, "seed": seed,
        "final_macro_f1": final.macro_f1,
        "wall_clock_seconds": elapsed,
        "n_fallback_rounds": n_fallback_rounds,
        "n_rounds": len(sim.gating_history),
    }


def main(n_max_per_class=3000, num_rounds=25):
    print(f"Loading N-BaIoT (n_max_per_class={n_max_per_class}) ...", flush=True)
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    results = []
    for seed in [1, 2]:
        out = run_one(clients, malicious, seed, num_rounds)
        results.append(out)
        with open(OUT_PATH, "w") as f:
            json.dump({
                "description": "N-BaIoT qwen2.5:14b hybrid, TB D=10, seeds 1-2 (seed=0 reference: 0.9974).",
                "reference_seed0": 0.9974,
                "runs": results,
            }, f, indent=2)

    print("\n=== Summary ===", flush=True)
    print("  seed=0 (reference): F1=0.9974")
    for r in results:
        if "error" in r:
            print(f"  seed={r['seed']}: FAILED")
        else:
            print(f"  seed={r['seed']}: F1={r['final_macro_f1']:.4f}")
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
