"""Calibration fix for the externalized-Bayesian hybrid policy
(Section VII-E TODO item 8): the uncalibrated version over-flagged
almost every round as anomalous for every client, driving all gating
weights to zero and silently degrading the whole run into plain FedAvg
via experiment.py's "never fully stall training" safety fallback. This
re-runs the identical D in {0, 10} trust-building experiment with
LLMExternalBayesianPolicy(calibrated=True), which adds three worked
few-shot examples (normal / borderline / clearly anomalous) to the
anomaly-judgment prompt so the model has a concrete reference scale.
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
        print(f"\n=== delay={delay} (model={MODEL}, calibrated hybrid) ===")
        policy = LLMExternalBayesianPolicy(model=MODEL, calibrated=True)
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
            results.append({"delay_rounds": delay, "strategy": f"gating_llm_hybrid_calibrated_{MODEL}",
                             "error": str(e)})
            continue
        elapsed = time.time() - t0
        final = logs[-1]
        row = {
            "delay_rounds": delay,
            "strategy": f"gating_llm_hybrid_calibrated_{MODEL}",
            "final_macro_f1": final.macro_f1,
            "wall_clock_seconds": elapsed,
            "sample_actions": final.actions,
            "sample_gating_scores": final.gating_scores,
        }
        results.append(row)
        print(f"  delay={delay:3d} strategy=gating_llm_hybrid_calibrated F1={final.macro_f1:.3f} ({elapsed:.1f}s)")
        path = os.path.join(RESULTS_DIR, "trustbuild_llm_gating_hybrid_calibrated_nbaiot.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> {path} (updated)")


if __name__ == "__main__":
    main()
