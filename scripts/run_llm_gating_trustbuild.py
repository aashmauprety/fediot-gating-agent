"""First real use of LLMGatingPolicy (local Ollama-backed) in an actual
experiment: repeats the trust-building attack comparison from
run_trustbuild_ablation.py, but with the real LLM-based Gating Agent in
place of the rule-based stand-in, at delays 0 and 10 (the two points
where the rule-based-vs-round-only gap was most dramatic: 0.320/0.174 at
D=0, 0.986/0.185 at D=10). This is the paper's single most consequential
open comparison: does actual LLM reasoning do better, worse, or about
the same as the fixed-threshold rule-based stand-in used everywhere else
in Section VII?

Much slower than the rule-based runs (~4.6s per gating decision vs.
~instant), so scoped to 2 of the 3 delays tested in the rule-based
version to keep wall-clock time reasonable (~15-20 min per delay).
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


def main(n_max_per_class=3000, num_rounds=25):
    print(f"Loading N-BaIoT (n_max_per_class={n_max_per_class}) ...")
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]
    delays = [0, 10]
    results = []
    for delay in delays:
        print(f"\n=== delay={delay} ===")
        policy = LLMGatingPolicy(model="llama3.1:8b")
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
            results.append({"delay_rounds": delay, "strategy": "gating_llm_ollama_llama3.1_8b",
                             "error": str(e)})
            continue
        elapsed = time.time() - t0
        final = logs[-1]
        row = {
            "delay_rounds": delay,
            "strategy": "gating_llm_ollama_llama3.1_8b",
            "final_macro_f1": final.macro_f1,
            "wall_clock_seconds": elapsed,
            "sample_actions": final.actions,
            "sample_gating_scores": final.gating_scores,
        }
        results.append(row)
        print(f"  delay={delay:3d} strategy=gating_llm F1={final.macro_f1:.3f} ({elapsed:.1f}s)")
        path = os.path.join(RESULTS_DIR, "trustbuild_llm_gating_nbaiot.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> {path} (updated)")


if __name__ == "__main__":
    main()
