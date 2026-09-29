"""Selective-LLM Gating Agent, matching the exact 5 N-BaIoT configurations
already reported for the full-LLM hybrid (trust_building D=0/10/20 plus
untargeted/targeted): does routing MOST clients through a cheap
deterministic z-score fast path, and calling the LLM only for the
minority the fast path is not confident about, preserve accuracy while
cutting the LLM-call count (and therefore the ~50-100x cost gap vs.
GShield documented in the paper)?

Reference numbers already on record for these exact configs
(qwen2.5:14b, full LLM, every round):
  trust_building D=0/10/20:  0.9958 / 0.9974 / 0.9961
  untargeted:                0.9958
  targeted:                  0.9972 (targeted_success_rate 0.0024)
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import SelectiveLLMGatingPolicy, LLMGatingPolicyError

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "nbaiot_selective_llm.json")
MODEL = "qwen2.5:14b"


def run_one(clients, malicious, attack, delay, num_rounds, seed):
    label = f"N-BaIoT, selective-LLM, attack={attack}" + (f", D={delay}" if delay is not None else "")
    print(f"\n=== {label} ===", flush=True)
    policy = SelectiveLLMGatingPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
        timeout_s=60.0, max_retries=3,
    )
    kwargs = dict(
        malicious_clients=malicious, attack=attack,
        gating=True, seed=seed, gating_policy=policy,
        trust_decay=0.9, force_accept_round0=True,
    )
    if delay is not None:
        kwargs["trust_building_delay"] = delay
    sim = FederatedSimulation(clients, num_classes=3, embed_dim=115, **kwargs)
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        print(f"FAILED: {e}", flush=True)
        return {"label": label, "attack": attack, "delay": delay, "error": str(e)}
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


def main(n_max_per_class=3000, num_rounds=25, seed=0):
    print(f"Loading N-BaIoT (n_max_per_class={n_max_per_class}) ...", flush=True)
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]
    print(f"{len(client_ids)} clients: {client_ids}", flush=True)

    configs = [
        ("trust_building", 0), ("trust_building", 10), ("trust_building", 20),
        ("untargeted", None), ("targeted", None),
    ]
    results = []
    for attack, delay in configs:
        out = run_one(clients, malicious, attack, delay, num_rounds, seed)
        results.append(out)
        with open(OUT_PATH, "w") as f:
            json.dump({
                "description": (
                    "N-BaIoT, selective-LLM Gating Agent: cheap deterministic "
                    "fast path for confidently-normal clients, real LLM call "
                    "only for the rest -- same 5 configs as the full-LLM "
                    "headline result, testing whether accuracy holds while "
                    "cutting the LLM-call count."
                ),
                "reference_full_llm": {
                    "trust_building_D0": 0.9958, "trust_building_D10": 0.9974,
                    "trust_building_D20": 0.9961, "untargeted": 0.9958,
                    "targeted": 0.9972,
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
