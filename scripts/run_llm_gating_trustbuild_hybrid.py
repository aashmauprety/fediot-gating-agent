"""Test of the externalized-Bayesian hybrid Gating Agent
(LLMExternalBayesianPolicy) against the same trust-building attack used
for every prior LLM-gating experiment. The LLM here only judges a single
round's digest in isolation (no posterior, no history); the long-term
trust override and the Bayesian posterior update are entirely external
and deterministic, identical in structure to RuleBasedGatingPolicy. This
directly targets the failure all three prior LLM configurations showed:
under-weighting long-term trust against a single anomalous round.

Same protocol as run_llm_gating_trustbuild.py: D in {0, 10}, 25 rounds,
9 real N-BaIoT clients, 3 malicious.
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
MODEL = "llama3.1:8b"


def main(n_max_per_class=3000, num_rounds=25):
    print(f"Loading N-BaIoT (n_max_per_class={n_max_per_class}) ...")
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]
    delays = [0, 10]
    results = []
    for delay in delays:
        print(f"\n=== delay={delay} (model={MODEL}, externalized-Bayesian hybrid) ===")
        policy = LLMExternalBayesianPolicy(model=MODEL)
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
            results.append({"delay_rounds": delay, "strategy": f"gating_llm_hybrid_{MODEL}",
                             "error": str(e)})
            continue
        elapsed = time.time() - t0
        final = logs[-1]
        row = {
            "delay_rounds": delay,
            "strategy": f"gating_llm_hybrid_{MODEL}",
            "final_macro_f1": final.macro_f1,
            "wall_clock_seconds": elapsed,
            "sample_actions": final.actions,
            "sample_gating_scores": final.gating_scores,
        }
        results.append(row)
        print(f"  delay={delay:3d} strategy=gating_llm_hybrid F1={final.macro_f1:.3f} ({elapsed:.1f}s)")
        path = os.path.join(RESULTS_DIR, "trustbuild_llm_gating_hybrid_nbaiot.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> {path} (updated)")


if __name__ == "__main__":
    main()
