"""Zero-recalibration cross-dataset transfer test: the paper's central
technical claim is that same-round, cross-client z-scoring makes the
digest deployment-agnostic -- no per-dataset threshold tuning needed.
This has never been tested directly: N-BaIoT and CIC IoT-DIAD were each
validated with the SAME fixed calibrated prompt, but that was never
framed as a deliberate transfer test.

This script is the sharp version: run the identical, unmodified
SelectiveLLMGatingPolicy (SYSTEM_PROMPT_CALIBRATED unchanged, same
few-shot examples built from CIC IoT-DIAD/N-BaIoT digest ranges) on CIC
IoT 2023 -- a real, different dataset never used for the Gating Agent
before, 39 features (vs. N-BaIoT's 115 and CIC IoT-DIAD's 118), 8
classes, real flow-level attack data. Zero threshold changes, zero
prompt edits, zero recalibration of any kind.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_ciciot2023_federated
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import SelectiveLLMGatingPolicy, LLMGatingPolicyError

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "ciciot2023_zero_recalibration.json")
MODEL = "qwen2.5:14b"


def run_one(clients, embed_dim, num_classes, malicious, attack, delay, num_rounds, seed):
    label = f"CIC IoT 2023, selective-LLM (zero recalibration), attack={attack}" + (f", D={delay}" if delay is not None else "")
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
    sim = FederatedSimulation(clients, num_classes=num_classes, embed_dim=embed_dim, **kwargs)
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        print(f"FAILED: {e}", flush=True)
        return {"label": label, "attack": attack, "delay": delay, "error": str(e)}
    elapsed = time.time() - t0
    final = logs[-1]
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    total = policy.n_fast_path + policy.n_llm_calls
    llm_frac = policy.n_llm_calls / total if total else 0.0
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
          f"fallback {n_fallback_rounds}/{len(sim.gating_history)}, "
          f"targeted_success={final.targeted_success_rate}, "
          f"LLM calls={llm_frac:.1%}", flush=True)
    return {
        "label": label, "attack": attack, "delay": delay,
        "final_macro_f1": final.macro_f1,
        "targeted_success_rate": final.targeted_success_rate,
        "wall_clock_seconds": elapsed,
        "n_fallback_rounds": n_fallback_rounds,
        "n_rounds": num_rounds,
        "llm_call_fraction": llm_frac,
    }


def main(n_per_class=3000, n_clients=9, num_rounds=25, seed=0):
    print("Loading real CIC IoT 2023 (never used for the Gating Agent before) ...", flush=True)
    clients = load_ciciot2023_federated(n_per_class=n_per_class, n_clients=n_clients, seed=seed)
    c0 = list(clients.values())[0]
    embed_dim = c0.X.shape[1]
    num_classes = len(c0.classes)
    print(f"{len(clients)} clients, embed_dim={embed_dim}, classes={c0.classes}", flush=True)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    configs = [
        ("trust_building", 10), ("untargeted", None), ("targeted", None),
    ]
    results = []
    for attack, delay in configs:
        out = run_one(clients, embed_dim, num_classes, malicious, attack, delay, num_rounds, seed)
        results.append(out)
        with open(OUT_PATH, "w") as f:
            json.dump({
                "description": (
                    "Zero-recalibration cross-dataset transfer: identical "
                    "SelectiveLLMGatingPolicy, unchanged prompt, on CIC IoT "
                    "2023 -- a real dataset never used to build or tune the "
                    "Gating Agent's few-shot calibration examples."
                ),
                "dataset": "CIC IoT 2023 (real flow features, 8 classes, 39 dims)",
                "embed_dim": embed_dim,
                "runs": results,
            }, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for r in results:
        if "error" in r:
            print(f"  {r['label']}: FAILED")
        else:
            print(f"  {r['label']}: F1={r['final_macro_f1']:.4f}")
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
