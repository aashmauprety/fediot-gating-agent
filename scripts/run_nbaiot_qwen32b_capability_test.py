"""Does a bigger local model help further, or is 14B already near a
capability ceiling for this task? Tests qwen2.5:32b on the identical
config already used for the 14B headline number (N-BaIoT, trust_building
D=10, seed=0, full LLM every round -- NOT the selective variant, for a
direct apples-to-apples comparison against the recorded 14B number:
0.9974 macro-F1, results/nbaiot_qwen14b_hybrid_fixed.json).
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
OUT_PATH = os.path.join(RESULTS_DIR, "nbaiot_qwen32b_capability_test.json")
MODEL = "qwen2.5:32b"


def main(n_max_per_class=3000, num_rounds=25, seed=0, delay=10):
    print(f"Loading N-BaIoT (n_max_per_class={n_max_per_class}) ...", flush=True)
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]
    print(f"{len(client_ids)} clients: {client_ids}", flush=True)

    policy = LLMExternalBayesianPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
        timeout_s=120.0, max_retries=3,
    )
    sim = FederatedSimulation(
        clients, num_classes=3, embed_dim=115,
        malicious_clients=malicious, attack="trust_building", trust_building_delay=delay,
        gating=True, seed=seed, gating_policy=policy,
        trust_decay=0.9, force_accept_round0=True,
    )
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        print(f"FAILED: {e}", flush=True)
        return
    elapsed = time.time() - t0
    final = logs[-1]
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
          f"fallback {n_fallback_rounds}/{len(sim.gating_history)} rounds", flush=True)
    print(f"Reference (qwen2.5:14b, identical config): 0.9974", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump({
            "description": (
                "qwen2.5:32b on the identical N-BaIoT TB D=10 seed=0 config "
                "used for the recorded 14B headline number (0.9974), full LLM "
                "every round -- tests whether 14B is near a capability ceiling."
            ),
            "model": MODEL,
            "reference_14b": 0.9974,
            "final_macro_f1": final.macro_f1,
            "wall_clock_seconds": elapsed,
            "n_fallback_rounds": n_fallback_rounds,
            "n_rounds": num_rounds,
        }, f, indent=2)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
