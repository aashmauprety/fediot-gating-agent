"""Phase 3 of the adaptive-attacker experiment (see
run_adaptive_evasion_gshield_sweep.py for Phase 1, run_adaptive_evasion_qwen14b.py
for Phase 2): does the constrain-and-scale attacker also defeat the NEW
attack surface introduced by the selective-LLM design specifically --
can blending toward its own clean-label gradient get a malicious
client's update INTO the free fast-path band (comfortably-normal digest,
no LLM call, no scrutiny at all), not just past GShield's clustering or
the full LLM's per-round judgment?

This matters because the fast path is a genuinely new mechanism (not
present in the full-LLM policy this defense is built from): if an
attacker can reliably land inside it, the LLM's rationale-bearing
judgment -- the paper's actual claimed advantage -- never gets a chance
to fire at all for that client, which would be a real, reportable
limitation of the cost-saving idea, not just a robustness curiosity.

Same lambda values as Phase 2 (0.0, 0.9, 0.99), same targeted attack,
same dataset, for direct comparison against the full-LLM reference
(0.9061-0.9068 range, essentially flat) and the GShield reference
(0.9207-0.9209, essentially flat).
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
OUT_PATH = os.path.join(RESULTS_DIR, "adaptive_evasion_selective_llm.json")
MODEL = "qwen2.5:14b"


def main(n_max_per_class=5000, n_clients=9, num_rounds=25, seed=0):
    print("Loading CIC IoT-DIAD 2024 attack data ...", flush=True)
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    results = []
    for lam in [0.0, 0.9, 0.99]:
        label = f"targeted, evasion_lambda={lam}, selective-LLM"
        print(f"\n=== {label} ===", flush=True)
        policy = SelectiveLLMGatingPolicy(
            model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
            timeout_s=60.0, max_retries=3,
        )
        sim = FederatedSimulation(
            clients, num_classes=len(classes), embed_dim=embed_dim,
            malicious_clients=malicious, attack="targeted",
            gating=True, seed=seed, gating_policy=policy,
            trust_decay=0.9, force_accept_round0=True,
            scale_factor=1.0, evasion_lambda=lam,
        )
        t0 = time.time()
        try:
            logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
        except LLMGatingPolicyError as e:
            print(f"FAILED: {e}", flush=True)
            results.append({"label": label, "evasion_lambda": lam, "error": str(e)})
            continue
        elapsed = time.time() - t0
        final = logs[-1]
        n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])

        # Fast-path bypass rate SPECIFICALLY for malicious clients: how
        # often did their (possibly blended) update land in the free
        # fast-path band, vs. trigger a real LLM call?
        malicious_fast = 0
        malicious_total = 0
        for h in sim.gating_history:
            for c in malicious:
                malicious_total += 1
                rationale = h["rationales"].get(c, "")
                if "fast path" in rationale:
                    malicious_fast += 1
        malicious_fast_frac = malicious_fast / malicious_total if malicious_total else 0.0

        total_decisions = policy.n_fast_path + policy.n_llm_calls
        overall_llm_frac = policy.n_llm_calls / total_decisions if total_decisions else 0.0

        print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
              f"fallback {n_fallback_rounds}/{len(sim.gating_history)}, "
              f"targeted_success={final.targeted_success_rate}, "
              f"malicious fast-path bypass={malicious_fast}/{malicious_total} "
              f"({malicious_fast_frac:.1%}), overall LLM-call frac={overall_llm_frac:.1%}", flush=True)
        for h in sim.gating_history[-5:]:
            m_actions = [h["actions"][c] for c in malicious]
            h_actions = [h["actions"][c] for c in client_ids if c not in malicious]
            print(f"  round {h['round']:2d}: malicious={m_actions}  honest={h_actions}", flush=True)
        results.append({
            "label": label, "evasion_lambda": lam,
            "final_macro_f1": final.macro_f1,
            "targeted_success_rate": final.targeted_success_rate,
            "wall_clock_seconds": elapsed,
            "n_fallback_rounds": n_fallback_rounds,
            "n_rounds": len(sim.gating_history),
            "malicious_fast_path_bypass_count": malicious_fast,
            "malicious_fast_path_bypass_total": malicious_total,
            "malicious_fast_path_bypass_fraction": malicious_fast_frac,
            "overall_llm_call_fraction": overall_llm_frac,
            "malicious_clients": malicious,
        })
        with open(OUT_PATH, "w") as f:
            json.dump({
                "description": (
                    "Selective-LLM Gating Agent under the constrain-and-scale "
                    "adaptive attack, targeted attack, CIC IoT-DIAD 2024. Tracks "
                    "whether the attacker can also land inside the free "
                    "fast-path band, not just evade the final decision. Compare "
                    "against the full-LLM reference (~0.906-0.907, flat across "
                    "lambda) and GShield reference (~0.9207-0.9209, flat)."
                ),
                "reference_full_llm": {"lambda_0.0": 0.9061, "lambda_0.9": None, "lambda_0.99": None},
                "reference_gshield": {"lambda_0.0": 0.9209, "lambda_0.9": 0.9208, "lambda_0.99": 0.9207},
                "runs": results,
            }, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for r in results:
        if "error" in r:
            print(f"  lambda={r['evasion_lambda']}: FAILED")
        else:
            print(f"  lambda={r['evasion_lambda']}: F1={r['final_macro_f1']:.4f}, "
                  f"targeted_success={r['targeted_success_rate']}, "
                  f"malicious_fast_path_bypass={r['malicious_fast_path_bypass_fraction']:.1%}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
