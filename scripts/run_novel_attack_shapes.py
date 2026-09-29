"""Novel attack-shape generalization test: does the Gating Agent detect
attack shapes it was never calibrated against, WITHOUT touching the
prompt at all (identical SYSTEM_PROMPT_CALIBRATED few-shot examples used
everywhere else in this paper)?

Three shapes, none resembling the untargeted/targeted/trust-building
signatures the calibration exemplars were built from:
  - slow_drip: attack strength ramps from 0 to full over 20 rounds
    instead of switching on at full strength
  - free_rider: not poisoning at all -- a client submits (almost) no
    real update, a distinct real FL threat category
  - intermittent: attacks only every 3rd round, honest otherwise

Also runs plain FedAvg (no defense) on the same three shapes, so we can
report the degradation each shape causes when undefended, exactly like
Table II/III's format for the existing attack types.
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
OUT_PATH = os.path.join(RESULTS_DIR, "novel_attack_shapes.json")
MODEL = "qwen2.5:14b"


def run_undefended(clients, malicious, attack, num_rounds, seed):
    sim = FederatedSimulation(
        clients, num_classes=3, embed_dim=115,
        malicious_clients=malicious, attack=attack, seed=seed,
        gating=False,
    )
    logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    return logs[-1].macro_f1


def run_gated(clients, malicious, attack, num_rounds, seed):
    policy = SelectiveLLMGatingPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
        timeout_s=60.0, max_retries=3,
    )
    sim = FederatedSimulation(
        clients, num_classes=3, embed_dim=115,
        malicious_clients=malicious, attack=attack,
        gating=True, seed=seed, gating_policy=policy,
        trust_decay=0.9, force_accept_round0=True,
    )
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        return {"error": str(e)}
    elapsed = time.time() - t0
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    total = policy.n_fast_path + policy.n_llm_calls
    return {
        "final_macro_f1": logs[-1].macro_f1,
        "wall_clock_seconds": elapsed,
        "n_fallback_rounds": n_fallback_rounds,
        "n_rounds": num_rounds,
        "llm_call_fraction": policy.n_llm_calls / total if total else 0.0,
    }


def main(n_max_per_class=3000, num_rounds=25, seed=0):
    print(f"Loading N-BaIoT (n_max_per_class={n_max_per_class}) ...", flush=True)
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]
    print(f"{len(client_ids)} clients: {client_ids}", flush=True)

    results = {}
    for attack in ["slow_drip", "free_rider", "intermittent"]:
        print(f"\n=== {attack}: undefended (plain FedAvg) ===", flush=True)
        undefended_f1 = run_undefended(clients, malicious, attack, num_rounds, seed)
        print(f"  F1={undefended_f1:.4f}", flush=True)

        print(f"=== {attack}: selective-LLM Gating Agent (UNCHANGED prompt) ===", flush=True)
        gated = run_gated(clients, malicious, attack, num_rounds, seed)
        if "error" in gated:
            print(f"  FAILED: {gated['error']}", flush=True)
        else:
            print(f"  F1={gated['final_macro_f1']:.4f} ({gated['wall_clock_seconds']:.1f}s), "
                  f"LLM calls={gated['llm_call_fraction']:.1%}", flush=True)

        results[attack] = {"undefended_macro_f1": undefended_f1, "gated": gated}
        with open(OUT_PATH, "w") as f:
            json.dump({
                "description": (
                    "Novel attack-shape generalization test: slow_drip, "
                    "free_rider, intermittent -- none resembling the "
                    "digest signatures the Gating Agent's few-shot "
                    "calibration examples were built from. Prompt was NOT "
                    "modified for this test."
                ),
                "results": results,
            }, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for attack, r in results.items():
        g = r["gated"]
        gf1 = g.get("final_macro_f1", "FAILED") if isinstance(g, dict) else g
        print(f"  {attack}: undefended={r['undefended_macro_f1']:.4f}, gated={gf1}", flush=True)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
