"""Human-evaluation pilot, step 1: regenerate the flagship Qwen2.5 14B
CIC IoT-DIAD D=10 run (identical config to run_ciciot_diad_hybrid_fixed_v3.py)
with a shadow AdaptiveRuleBasedGatingPolicy running alongside the real
LLM policy on every round's IDENTICAL digest (never consulted for the
real gating decision or aggregation weights -- see
fedgate/experiment.py's `shadow_policy`). This produces, for every real
round/client in the flagship run, both the LLM's actual rationale and
the deterministic rule-based policy's rationale for the exact same
underlying event -- the paired material needed for a blind human
comparison (reviewer-flagged weakness: "auditability... not yet
validated with human evaluators").

This is a fresh LLM run, not a replay of the original
ciciot_diad_hybrid_d10_fixes12345_qwen14b.json (which didn't log
digests), so expect the accuracy number to land in the same tight band
as the original (0.896-0.912 across seeds) but not be bit-identical --
real LLM sampling/timing can differ run to run even at temperature 0
due to Ollama server state. Report the number this run actually gets,
not the original.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_ciciot_diad_attack_classification
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import LLMExternalBayesianPolicy, LLMGatingPolicyError, AdaptiveRuleBasedGatingPolicy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
MODEL = "qwen2.5:14b"


def main(n_max_per_class=5000, n_clients=9, num_rounds=25, delay=10, seed=0):
    print("Loading CIC IoT-DIAD 2024 attack data ...", flush=True)
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    policy = LLMExternalBayesianPolicy(model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5)
    shadow_policy = AdaptiveRuleBasedGatingPolicy(warmup_rounds=5, warmup_scale=1.5)

    sim = FederatedSimulation(
        clients, num_classes=len(classes), embed_dim=embed_dim,
        malicious_clients=malicious, attack="trust_building",
        gating=True, trust_building_delay=delay, seed=seed,
        gating_policy=policy, shadow_policy=shadow_policy,
        trust_decay=0.9, force_accept_round0=True,
    )
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        print(f"FAILED: {e}", flush=True)
        return
    elapsed = time.time() - t0
    final = logs[-1]
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
          f"fallback {n_fallback_rounds}/{len(sim.gating_history)} rounds", flush=True)

    # Quick agreement check: how often do the real (LLM) and shadow
    # (rule-based) actions actually match, for honesty about how much
    # the pilot is really testing "different explanation, same verdict"
    # vs "different explanation, different verdict too".
    n_total, n_agree = 0, 0
    for h in sim.gating_history:
        for cid in client_ids:
            n_total += 1
            if h["actions"][cid] == h["shadow_actions"][cid]:
                n_agree += 1
    print(f"Action agreement (LLM vs shadow rule-based): {n_agree}/{n_total} ({n_agree/n_total:.1%})", flush=True)

    out = {
        "description": (
            f"CIC IoT-DIAD 2024, D={delay}, model={MODEL}, with a shadow "
            f"AdaptiveRuleBasedGatingPolicy logged alongside the real LLM "
            f"policy on the same digest every round -- source data for the "
            f"human-evaluation pilot (LLM rationale vs templated rule-based "
            f"rationale, same underlying event)."
        ),
        "delay": delay,
        "model": MODEL,
        "final_macro_f1": final.macro_f1,
        "wall_clock_seconds": elapsed,
        "n_fallback_rounds": n_fallback_rounds,
        "n_rounds": len(sim.gating_history),
        "malicious_clients": malicious,
        "action_agreement_fraction": n_agree / n_total,
        "gating_history": sim.gating_history,
    }
    path = os.path.join(RESULTS_DIR, "human_eval_pilot_source_run.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}", flush=True)


if __name__ == "__main__":
    main()
