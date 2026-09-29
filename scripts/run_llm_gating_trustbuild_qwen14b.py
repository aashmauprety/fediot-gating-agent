"""Same trust-building comparison as run_llm_gating_trustbuild.py
(single-turn LLMGatingPolicy, no FetchHistory tool), but with a bigger
local model (qwen2.5:14b instead of llama3.1:8b) to test whether the
D=10 collapse seen with the 8B model (0.251 macro-F1, worse than plain
FedAvg) is a model-capacity limitation rather than something inherent to
local/single-turn LLM gating.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import LLMGatingPolicy, LLMGatingPolicyError

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL = "qwen2.5:14b"


def main(n_max_per_class=3000, num_rounds=25):
    print(f"Loading N-BaIoT (n_max_per_class={n_max_per_class}) ...")
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]
    delays = [0, 10]
    results = []
    for delay in delays:
        print(f"\n=== delay={delay} (model={MODEL}) ===")
        policy = LLMGatingPolicy(model=MODEL)
        sim = FederatedSimulation(
            clients, num_classes=3, embed_dim=115,
            malicious_clients=malicious, attack="trust_building",
            gating=True, trust_building_delay=delay, seed=0,
            gating_policy=policy,
        )
        t0 = time.time()
        try:
            logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
        except LLMGatingPolicyError as e:
            print(f"  FAILED at delay={delay}: {e}")
            results.append({"delay_rounds": delay, "strategy": f"gating_llm_ollama_{MODEL}",
                             "error": str(e)})
            continue
        elapsed = time.time() - t0
        final = logs[-1]
        row = {
            "delay_rounds": delay,
            "strategy": f"gating_llm_ollama_{MODEL}",
            "final_macro_f1": final.macro_f1,
            "wall_clock_seconds": elapsed,
            "sample_actions": final.actions,
            "sample_gating_scores": final.gating_scores,
        }
        results.append(row)
        print(f"  delay={delay:3d} strategy=gating_llm_qwen14b F1={final.macro_f1:.3f} ({elapsed:.1f}s)")
        path = os.path.join(RESULTS_DIR, "trustbuild_llm_gating_qwen14b_nbaiot.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> {path} (updated)")


if __name__ == "__main__":
    main()
