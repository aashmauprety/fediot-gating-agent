"""Third iteration on the Section VII-E hybrid Gating Agent fix (see
README "Ninth"/"Tenth real result"). Fixes #1-3 (round-0 force-accept,
early-round warmup thresholds, trust-posterior decay) bought 0.241 ->
0.243. Fixes #4-5 (real calib_loss_delta, textual round-aware cosine
caveat) on top of those made things WORSE (0.186) -- the 8B model kept
judging raw cosine against a fixed anchor regardless of the caveat, and
sometimes read calib_loss_delta's sign backwards.

This script replaces both broken pieces with same-round cross-client
z-scoring (cosine handled exactly like update_norm already was, so a
round-wide shift cancels out structurally instead of needing a prose
caveat) and a worded calib_loss_delta direction (so sign can't be
misread) -- see gating_agent.py's rewritten SYSTEM_PROMPT_CALIBRATED
and _build_prompt. Ad hoc synthetic smoke tests before this run showed
llama3.1:8b *still* misjudges these z-scores (comparing the two
z-scores to each other rather than each to its own scale), but
qwen2.5:14b handles all five hand-built cases correctly (accepts a
truly normal case, downweights genuine borderline cases, excludes a
truly extreme case, and -- critically -- does NOT treat a large
*negative* calib_loss_delta, i.e. an improving loss, as suspicious).
So this run uses qwen2.5:14b, not llama3.1:8b.
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
MODEL = "qwen2.5:14b"


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

    label = f"fixes #1-5, {MODEL}, z-scored cosine + worded calib_loss_delta"
    print(f"\n=== {label} ===")
    policy = LLMExternalBayesianPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
    )
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
        print(f"FAILED: {e}")
        return
    elapsed = time.time() - t0
    final = logs[-1]
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
          f"fallback {n_fallback_rounds}/{len(sim.gating_history)} rounds")
    print("Per-round action summary (M=malicious, H=honest):")
    for h in sim.gating_history:
        m_actions = [h["actions"][c] for c in malicious]
        h_actions = [h["actions"][c] for c in client_ids if c not in malicious]
        print(f"  round {h['round']:2d}: malicious={m_actions}  honest={h_actions}")

    print("\n=== Comparison ===")
    print("  rule-based reference (upper bound): 0.758")
    print("  plain single-turn LLM (8B): 0.064")
    print("  calibrated hybrid (8B), no fixes: 0.2412")
    print("  fixes #1-3 (8B): 0.2430")
    print("  fixes #1-3 + raw calib_loss_delta + textual cosine caveat (8B): 0.1863")
    print(f"  fixes #1-5, z-scored cosine + worded calib delta ({MODEL}): {final.macro_f1:.4f}")

    out = {
        "description": (
            f"CIC IoT-DIAD 2024, calibrated externalized-Bayesian hybrid Gating "
            f"Agent, D=10, model={MODEL} -- fixes #1-3 plus same-round "
            f"cross-client z-scored cosine and worded calib_loss_delta (#4-5 v2)."
        ),
        "delay": delay,
        "model": MODEL,
        "reference_rule_based_f1": 0.758,
        "reference_plain_llm_8b_f1": 0.064,
        "reference_fixes123_8b_f1": 0.2430,
        "reference_fixes12345_8b_f1": 0.1863,
        "final_macro_f1": final.macro_f1,
        "wall_clock_seconds": elapsed,
        "n_fallback_rounds": n_fallback_rounds,
        "n_rounds": len(sim.gating_history),
        "malicious_clients": malicious,
        "gating_history": sim.gating_history,
    }
    path = os.path.join(RESULTS_DIR, f"ciciot_diad_hybrid_d{delay}_fixes12345_qwen14b.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
