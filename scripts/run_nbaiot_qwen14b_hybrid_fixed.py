"""Best-next-task per user request 2026-09-26: generalize the winning
CIC IoT-DIAD 2024 result (README "eleventh/twelfth real result": qwen2.5:14b
+ hybrid architecture + z-scored cosine + worded calib_loss_delta + fixes
#1/#3, F1=0.90+ beating the rule-based reference of 0.758) to a SECOND
dataset, N-BaIoT -- this dataset is where the original, still-unexplained
"N-BaIoT calibration-regression mystery" (README Section VII-E TODO item,
8B calibrated hybrid collapsing to ~0.18 at both D=0 and D=10) was first
found, and it was never re-tried with the fixes that later worked on
CIC IoT-DIAD.

Reference numbers already on record for this exact N-BaIoT setup
(9 N-BaIoT devices as clients, malicious=first 3, trust_building attack):
  - rule-based policy, WITH Bayesian long-term override
    (results/trustbuild_ablation_nbaiot.json, strategy=gating_with_bayesian):
      D=0:  0.320   D=10: 0.986   D=20: 0.988
  - qwen2.5:14b, single-turn (non-hybrid), OLD prompt
    (results/trustbuild_llm_gating_qwen14b_nbaiot.json):
      D=0:  0.996   D=10: 0.386   (D=20 not run)
  - llama3.1:8b, hybrid, OLD prompt, calibrated
    (results/trustbuild_llm_gating_hybrid_calibrated_nbaiot.json):
      D=0:  0.185   D=10: 0.174   (D=20 not run) -- this is "the mystery"

This script runs qwen2.5:14b + hybrid architecture + the full fix set
(z-scored cosine, worded calib_loss_delta, force_accept_round0,
warmup_rounds=5, trust_decay=0.9) at D=0, D=10, and D=20, to see whether
the CIC IoT-DIAD success generalizes to this second, structurally
different dataset (9 real N-BaIoT devices vs. 9 synthetic client splits
of one CIC IoT-DIAD pool).
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
OUT_PATH = os.path.join(RESULTS_DIR, "nbaiot_qwen14b_hybrid_fixed.json")
MODEL = "qwen2.5:14b"


def run_one(clients, malicious, delay, num_rounds, seed):
    label = f"N-BaIoT, qwen2.5:14b hybrid + fixes, D={delay}"
    print(f"\n=== {label} ===", flush=True)
    policy = LLMExternalBayesianPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
        timeout_s=60.0, max_retries=3,
    )
    sim = FederatedSimulation(
        clients, num_classes=3, embed_dim=115,
        malicious_clients=malicious, attack="trust_building",
        gating=True, trust_building_delay=delay, seed=seed,
        gating_policy=policy, trust_decay=0.9, force_accept_round0=True,
    )
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        print(f"FAILED: {e}", flush=True)
        return {"label": label, "delay": delay, "error": str(e)}
    elapsed = time.time() - t0
    final = logs[-1]
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    client_ids = list(clients.keys())
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
          f"fallback {n_fallback_rounds}/{len(sim.gating_history)} rounds", flush=True)
    for h in sim.gating_history[-5:]:
        m_actions = [h["actions"][c] for c in malicious]
        h_actions = [h["actions"][c] for c in client_ids if c not in malicious]
        print(f"  round {h['round']:2d}: malicious={m_actions}  honest={h_actions}", flush=True)
    return {
        "label": label,
        "delay": delay,
        "final_macro_f1": final.macro_f1,
        "wall_clock_seconds": elapsed,
        "n_fallback_rounds": n_fallback_rounds,
        "n_rounds": len(sim.gating_history),
        "malicious_clients": malicious,
        "gating_history": sim.gating_history,
    }


def main(n_max_per_class=3000, num_rounds=25, seed=0):
    print(f"Loading N-BaIoT (n_max_per_class={n_max_per_class}) ...", flush=True)
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]
    print(f"{len(client_ids)} clients: {client_ids}", flush=True)

    delays = [0, 10, 20]
    results = []
    for delay in delays:
        out = run_one(clients, malicious, delay, num_rounds, seed)
        results.append(out)
        with open(OUT_PATH, "w") as f:
            json.dump({
                "description": (
                    "N-BaIoT, qwen2.5:14b hybrid Gating Agent with the full "
                    "fix set (z-scored cosine, worded calib_loss_delta, "
                    "force_accept_round0, warmup, trust_decay) -- generalizing "
                    "the CIC IoT-DIAD 2024 result to a second dataset."
                ),
                "reference_rule_based_bayesian": {"D=0": 0.3204, "D=10": 0.9858, "D=20": 0.9880},
                "reference_qwen14b_singleturn_old_prompt": {"D=0": 0.9963, "D=10": 0.3860},
                "reference_8b_hybrid_old_prompt_calibrated": {"D=0": 0.1846, "D=10": 0.1742},
                "runs": results,
            }, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for r in results:
        if "error" in r:
            print(f"  {r['label']}: FAILED", flush=True)
        else:
            print(f"  {r['label']}: F1={r['final_macro_f1']:.4f}, "
                  f"fallback {r['n_fallback_rounds']}/{r['n_rounds']}", flush=True)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
