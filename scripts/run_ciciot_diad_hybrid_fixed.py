"""Section VII-E TODO item 1: test the three independently addressable
fixes diagnosed from the calibrated-hybrid D=10 failure
(results/ciciot_diad_hybrid_d10_calibratedTrue.json, F1=0.241, fallback
21/25 rounds), applied together on the exact same setup:

1. Round 0 is a non-representative, code-forced easy case (cosine
   hardcoded to 1.0, no real prior global update to compare against) --
   `FederatedSimulation(force_accept_round0=True)` (now the default):
   round 0 is accepted for every client without consulting the policy or
   feeding the trust posterior, so it can no longer masquerade as either
   real calibration success or a data point.
2. Early-training digests (rounds ~1-5) are genuinely more volatile --
   `LLMExternalBayesianPolicy(warmup_rounds=5, warmup_scale=1.5)` widens
   the exclude/downweight anomaly thresholds for the first 5 rounds,
   linearly decaying back to the normal threshold by round 5.
3. The trust posterior had no recovery path -- `BayesianTrust(decay=...)`
   via `FederatedSimulation(trust_decay=0.9)` decays old (alpha, beta)
   mass toward the Beta(1,1) prior each update, so one wrongly-excluded
   early round no longer permanently caps how high the posterior mean
   can climb.

Reference numbers this compares against (same D=10, same seed, same
calibrated-hybrid policy, no fixes applied):
  - rule-based (not LLM, upper reference): 0.758
  - plain single-turn LLM (no hybrid): 0.064
  - calibrated hybrid, NO fixes (this script's baseline): 0.241
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


def run_one(clients, classes, embed_dim, malicious, delay, num_rounds, seed,
            warmup_rounds, warmup_scale, trust_decay, force_accept_round0, label):
    print(f"\n=== {label} ===")
    policy = LLMExternalBayesianPolicy(
        model=MODEL, calibrated=True,
        warmup_rounds=warmup_rounds, warmup_scale=warmup_scale,
    )
    sim = FederatedSimulation(
        clients, num_classes=len(classes), embed_dim=embed_dim,
        malicious_clients=malicious, attack="trust_building",
        gating=True, trust_building_delay=delay, seed=seed,
        gating_policy=policy,
        trust_decay=trust_decay,
        force_accept_round0=force_accept_round0,
    )
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        print(f"FAILED: {e}")
        return None
    elapsed = time.time() - t0
    final = logs[-1]
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
          f"fallback {n_fallback_rounds}/{len(sim.gating_history)} rounds")
    print("Per-round action summary (M=malicious, H=honest):")
    client_ids = list(clients.keys())
    for h in sim.gating_history:
        m_actions = [h["actions"][c] for c in malicious]
        h_actions = [h["actions"][c] for c in client_ids if c not in malicious]
        print(f"  round {h['round']:2d}: malicious={m_actions}  honest={h_actions}")
    return {
        "label": label,
        "final_macro_f1": final.macro_f1,
        "wall_clock_seconds": elapsed,
        "n_fallback_rounds": n_fallback_rounds,
        "n_rounds": len(sim.gating_history),
        "gating_history": sim.gating_history,
    }


def main(n_max_per_class=5000, n_clients=9, num_rounds=25, delay=10, seed=0):
    print("Loading CIC IoT-DIAD 2024 attack data ...")
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    print(f"{len(classes)} classes, embed_dim={embed_dim}, {len(clients)} clients")

    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    runs = [
        dict(warmup_rounds=0, warmup_scale=1.0, trust_decay=1.0,
             force_accept_round0=False, label="baseline (no fixes, reproduces 0.241)"),
        dict(warmup_rounds=0, warmup_scale=1.0, trust_decay=1.0,
             force_accept_round0=True, label="fix #1 only: force-accept round 0"),
        dict(warmup_rounds=5, warmup_scale=1.5, trust_decay=1.0,
             force_accept_round0=True, label="fix #1 + #2: + early-round warmup thresholds"),
        dict(warmup_rounds=5, warmup_scale=1.5, trust_decay=0.9,
             force_accept_round0=True, label="fix #1 + #2 + #3: + trust decay=0.9"),
    ]

    results = []
    for cfg in runs:
        label = cfg.pop("label")
        out = run_one(clients, classes, embed_dim, malicious, delay, num_rounds, seed, label=label, **cfg)
        if out is not None:
            results.append(out)

    print("\n=== Summary ===")
    print("Reference: rule-based D=10 = 0.758, plain single-turn LLM D=10 = 0.064")
    for r in results:
        print(f"  {r['label']}: F1={r['final_macro_f1']:.4f}, "
              f"fallback {r['n_fallback_rounds']}/{r['n_rounds']}")

    out = {
        "description": (
            "CIC IoT-DIAD 2024, calibrated externalized-Bayesian hybrid Gating "
            "Agent, D=10 -- ablating the three fixes for the diagnosed "
            "round-0-artifact / early-round-volatility / no-trust-recovery "
            "failure (Section VII-E TODO item 1)."
        ),
        "delay": delay,
        "reference_rule_based_f1": 0.758,
        "reference_plain_llm_f1": 0.064,
        "runs": results,
    }
    path = os.path.join(RESULTS_DIR, f"ciciot_diad_hybrid_d{delay}_fixes_ablation.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
