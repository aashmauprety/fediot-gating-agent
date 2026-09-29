"""Real test of the externalized-Bayesian hybrid Gating Agent
(LLMExternalBayesianPolicy) on CIC IoT-DIAD 2024's richer feature set,
WITH full per-round logging this time (experiment.py's new
gating_history, added specifically because its absence made the
N-BaIoT calibration-fix regression unexplainable -- Section VII-E TODO
items 9-10). This is the first real experiment to use that logging.

Focused on D=10 specifically first (the case both the rule-based
reference, 0.758, and the plain single-turn LLM, 0.064, most sharply
diverge on) rather than repeating the full D-sweep, to get a real,
fully-diagnosed answer faster.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_ciciot_diad_attack_classification
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import LLMExternalBayesianPolicy, LLMGatingPolicyError

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL = "llama3.1:8b"


def main(n_max_per_class=5000, n_clients=9, num_rounds=25, delay=10, calibrated=False):
    print(f"Loading CIC IoT-DIAD 2024 attack data ...")
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    print(f"{len(classes)} classes, embed_dim={embed_dim}, {len(clients)} clients")

    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    print(f"\n=== delay={delay} (model={MODEL}, externalized-Bayesian hybrid, "
          f"calibrated={calibrated}, WITH per-round logging) ===")
    policy = LLMExternalBayesianPolicy(model=MODEL, calibrated=calibrated)
    sim = FederatedSimulation(
        clients, num_classes=len(classes), embed_dim=embed_dim,
        malicious_clients=malicious, attack="trust_building",
        gating=True, trust_building_delay=delay, seed=0,
        gating_policy=policy,
    )
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        print(f"FAILED: {e}")
        return
    elapsed = time.time() - t0
    final = logs[-1]
    print(f"\nFinal macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s)")
    print(f"Reference: rule-based D=10 = 0.758, plain single-turn LLM D=10 = 0.064")

    # The actual point of this run: real per-round diagnostics.
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    print(f"\nFallback (all-weights-zero) triggered in {n_fallback_rounds}/{len(sim.gating_history)} rounds")

    # How did actions evolve for malicious vs. honest clients over time?
    print("\nPer-round action summary (M=malicious, H=honest):")
    for h in sim.gating_history:
        m_actions = [h["actions"][c] for c in malicious]
        h_actions = [h["actions"][c] for c in client_ids if c not in malicious]
        print(f"  round {h['round']:2d}: malicious={m_actions}  honest={h_actions}")

    out = {
        "description": "CIC IoT-DIAD 2024, externalized-Bayesian hybrid Gating Agent, WITH full per-round logging",
        "delay": delay,
        "calibrated": calibrated,
        "final_macro_f1": final.macro_f1,
        "wall_clock_seconds": elapsed,
        "malicious_clients": malicious,
        "n_fallback_rounds": n_fallback_rounds,
        "gating_history": sim.gating_history,
    }
    path = os.path.join(RESULTS_DIR, f"ciciot_diad_hybrid_d{delay}_calibrated{calibrated}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
