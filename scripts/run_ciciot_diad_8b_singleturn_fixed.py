"""User request: try the 8B single-turn (non-hybrid) Gating Agent
architecture with the same prompt fix that rescued the hybrid
architecture (README "eleventh/twelfth real result"): cosine as a
same-round cross-client z-score instead of a raw value, and a worded
calib_loss_delta direction instead of a signed number. Applied to
`LLMGatingPolicy` (both SYSTEM_PROMPT_NO_TOOL and SYSTEM_PROMPT_WITH_TOOL,
and `_format_history` for the FetchHistory variant) in
`fedgate/gating_agent.py`.

Note on scope: fix #1 (force_accept_round0) and fix #3 (trust_decay)
apply here unchanged, since they live in FederatedSimulation/
BayesianTrust, outside any specific policy. Fix #2 (early-round warmup,
which widened a fixed numeric exclude/downweight threshold) does NOT
apply to LLMGatingPolicy -- this policy has no such threshold; the
model decides the action directly. Before running the full experiment,
synthetic smoke tests (4 hand-built digests, same as used to vet the
hybrid fix) showed llama3.1:8b still fails on this single-turn
architecture even with the new prompt -- it downweighted a "clearly
normal" case in this exact configuration, unlike qwen2.5:14b (also
smoke-tested) which correctly accepted it and only differed on
downweighting a brand-new gateway with no history yet purely due to
lack of observations, a more defensible caution rather than a numeric
misjudgment. This full run's job is to confirm (or overturn) that
smoke-test signal, not just repeat the synthetic sanity check.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_ciciot_diad_attack_classification
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import LLMGatingPolicy, LLMGatingPolicyError

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")


def run_one(clients, classes, embed_dim, malicious, delay, num_rounds, seed, model, use_history_tool, label):
    print(f"\n=== {label} ===", flush=True)
    policy = LLMGatingPolicy(model=model, use_history_tool=use_history_tool, timeout_s=60.0, max_retries=3)
    sim = FederatedSimulation(
        clients, num_classes=len(classes), embed_dim=embed_dim,
        malicious_clients=malicious, attack="trust_building",
        gating=True, trust_building_delay=delay, seed=seed,
        gating_policy=policy, trust_decay=0.9, force_accept_round0=True,
    )
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        print(f"FAILED: {e}", flush=True)
        return {"label": label, "error": str(e)}
    elapsed = time.time() - t0
    final = logs[-1]
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    client_ids = list(clients.keys())
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
          f"fallback {n_fallback_rounds}/{len(sim.gating_history)} rounds", flush=True)
    for h in sim.gating_history[-6:]:
        m_actions = [h["actions"][c] for c in malicious]
        h_actions = [h["actions"][c] for c in client_ids if c not in malicious]
        print(f"  round {h['round']:2d}: malicious={m_actions}  honest={h_actions}", flush=True)
    return {
        "label": label,
        "model": model,
        "use_history_tool": use_history_tool,
        "final_macro_f1": final.macro_f1,
        "wall_clock_seconds": elapsed,
        "n_fallback_rounds": n_fallback_rounds,
        "n_rounds": len(sim.gating_history),
        "malicious_clients": malicious,
        "gating_history": sim.gating_history,
    }


def main(n_max_per_class=5000, n_clients=9, num_rounds=25, delay=10, seed=0):
    print("Loading CIC IoT-DIAD 2024 attack data ...", flush=True)
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    print(f"{len(classes)} classes, embed_dim={embed_dim}, {len(clients)} clients", flush=True)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    result = run_one(
        clients, classes, embed_dim, malicious, delay, num_rounds, seed,
        model="llama3.1:8b", use_history_tool=False,
        label="8B single-turn, z-scored cosine + worded calib delta, D=10 seed=0",
    )

    print("\n=== Comparison ===")
    print("  rule-based reference: 0.758")
    print("  8B single-turn, OLD prompt (raw cosine, signed calib_delta): 0.064")
    print("  8B hybrid, OLD prompt: 0.241")
    print("  8B hybrid, NEW prompt (fixes #1-5 v2): 0.186 (regressed)")
    print("  qwen2.5:14b hybrid, NEW prompt: 0.904-0.912 (fixed)")
    if "error" not in result:
        print(f"  8B single-turn, NEW prompt (this run): {result['final_macro_f1']:.4f}")
    else:
        print(f"  8B single-turn, NEW prompt (this run): FAILED")

    path = os.path.join(RESULTS_DIR, f"ciciot_diad_8b_singleturn_newprompt_d{delay}.json")
    with open(path, "w") as f:
        json.dump({
            "description": (
                "CIC IoT-DIAD 2024, 8B single-turn (non-hybrid) LLMGatingPolicy "
                "with the z-scored-cosine + worded-calib-delta prompt fix, D=10."
            ),
            "reference_rule_based_f1": 0.758,
            "reference_8b_singleturn_old_prompt_f1": 0.064,
            "reference_8b_hybrid_old_prompt_f1": 0.241,
            "reference_8b_hybrid_new_prompt_f1": 0.186,
            "reference_qwen14b_hybrid_new_prompt_f1": 0.9041,
            "result": result,
        }, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
