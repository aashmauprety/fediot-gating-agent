"""Selective-LLM Gating Agent on the SECOND dataset, CIC IoT-DIAD 2024,
matching the exact 6 configurations already reported for the full-LLM
hybrid (Table II of the paper): does the ~48% LLM-call reduction found
on N-BaIoT (accuracy preserved, see scripts/run_nbaiot_selective_llm.py)
generalize to this structurally different dataset too, or was it an
N-BaIoT-specific artifact?

Reference numbers already on record for these exact configs
(qwen2.5:14b, full LLM, every round):
  trust_building D=10 (3 seeds): 0.896-0.912
  trust_building D=20:           0.898
  untargeted:                    0.912
  targeted:                      0.907
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_ciciot_diad_attack_classification
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import SelectiveLLMGatingPolicy, LLMGatingPolicyError

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "ciciot_diad_selective_llm.json")
MODEL = "qwen2.5:14b"


def run_one(clients, classes, embed_dim, malicious, attack, delay, num_rounds, seed, label):
    print(f"\n=== {label} ===", flush=True)
    policy = SelectiveLLMGatingPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
    )
    kwargs = dict(
        clients=clients, num_classes=len(classes), embed_dim=embed_dim,
        malicious_clients=malicious, attack=attack,
        gating=True, seed=seed, gating_policy=policy,
        trust_decay=0.9, force_accept_round0=True,
    )
    if attack == "trust_building":
        kwargs["trust_building_delay"] = delay
    sim = FederatedSimulation(**kwargs)
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        print(f"FAILED: {e}", flush=True)
        return {"label": label, "attack": attack, "delay": delay, "seed": seed, "error": str(e)}
    elapsed = time.time() - t0
    final = logs[-1]
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    total_decisions = policy.n_fast_path + policy.n_llm_calls
    llm_call_frac = policy.n_llm_calls / total_decisions if total_decisions else 0.0
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
          f"fallback {n_fallback_rounds}/{len(sim.gating_history)} rounds, "
          f"LLM calls {policy.n_llm_calls}/{total_decisions} ({llm_call_frac:.1%})", flush=True)
    return {
        "label": label,
        "attack": attack,
        "delay": delay,
        "seed": seed,
        "final_macro_f1": final.macro_f1,
        "targeted_success_rate": final.targeted_success_rate,
        "wall_clock_seconds": elapsed,
        "n_fallback_rounds": n_fallback_rounds,
        "n_rounds": len(sim.gating_history),
        "n_fast_path": policy.n_fast_path,
        "n_llm_calls": policy.n_llm_calls,
        "llm_call_fraction": llm_call_frac,
        "malicious_clients": malicious,
    }


def main(n_max_per_class=5000, n_clients=9, num_rounds=25):
    print("Loading CIC IoT-DIAD 2024 attack data ...", flush=True)
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    print(f"{len(classes)} classes, embed_dim={embed_dim}, {len(clients)} clients", flush=True)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    runs_spec = [
        ("trust_building", 10, 0, "trust_building D=10 seed=0"),
        ("trust_building", 10, 1, "trust_building D=10 seed=1"),
        ("trust_building", 10, 2, "trust_building D=10 seed=2"),
        ("trust_building", 20, 0, "trust_building D=20 seed=0"),
        ("untargeted", 0, 0, "untargeted seed=0"),
        ("targeted", 0, 0, "targeted seed=0"),
    ]

    results = []
    for attack, delay, seed, label in runs_spec:
        out = run_one(clients, classes, embed_dim, malicious, attack, delay, num_rounds, seed, label)
        results.append(out)
        with open(OUT_PATH, "w") as f:
            json.dump({
                "description": (
                    "CIC IoT-DIAD 2024, selective-LLM Gating Agent, matching "
                    "the 6 configs already reported for the full-LLM hybrid, "
                    "testing whether the N-BaIoT LLM-call reduction generalizes."
                ),
                "reference_full_llm": {
                    "trust_building_D10_seed0": 0.896, "trust_building_D10_seed1": 0.912,
                    "trust_building_D10_seed2": 0.905, "trust_building_D20": 0.898,
                    "untargeted": 0.912, "targeted": 0.907,
                },
                "runs": results,
            }, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for r in results:
        if "error" in r:
            print(f"  {r['label']}: FAILED", flush=True)
        else:
            print(f"  {r['label']}: F1={r['final_macro_f1']:.4f}, "
                  f"LLM calls={r['n_llm_calls']}/{r['n_fast_path']+r['n_llm_calls']} "
                  f"({r['llm_call_fraction']:.1%})", flush=True)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
