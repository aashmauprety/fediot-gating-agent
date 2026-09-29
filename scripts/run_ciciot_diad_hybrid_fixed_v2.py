"""Follow-up to run_ciciot_diad_hybrid_fixed.py (README "Ninth real
result"): fixes #1-#3 (round-0 force-accept, early-round warmup
thresholds, trust-posterior decay) only bought 0.241 -> 0.287, and
inspecting gating_history showed the LLM had stopped discriminating
malicious from honest clients at all from round ~4 onward, for two
reasons not covered by fixes #1-#3:

4. `calib_loss_delta` was a hardcoded `0.0` in every Digest (never
   actually computed) -- fixed in `fedgate/experiment.py`:
   `FederatedSimulation._calib_loss` now evaluates real held-out
   cross-entropy loss (on a fixed calibration set built from each
   client's val split) before and after each client's local update, so
   `calib_loss_delta` is a real signal whenever `gating=True`.
5. The calibrated system prompt's exemplars anchored "normal" cosine
   similarity at ~0.94 with no round-awareness, so late-training
   honest clients (whose cosine naturally drifts lower/noisier as the
   global model converges) got misread as attacks --
   `LLMExternalBayesianPolicy`'s calibrated system prompt now includes
   an explicit caveat about this, and `_build_prompt` now tells the
   model which training round it is.

This script re-runs the exact same D=10, calibrated-hybrid setup with
fixes #1+#2+#3 already applied (since #4 and #5 are now unconditionally
in effect whenever gating=True / calibrated=True, there is nothing to
toggle for them -- this is simply "fixes #1+#2+#3, on the current code")
to see whether closing the two newly-diagnosed information gaps
recovers more of the gap to the rule-based reference (0.758).
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

    label = "fix #1+2+3 + real calib_loss_delta + round-aware cosine prompt"
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
    print("  plain single-turn LLM: 0.064")
    print("  calibrated hybrid, no fixes: 0.2412")
    print("  fixes #1+2+3 only (no calib_loss_delta / cosine prompt fix): 0.2430")
    print(f"  fixes #1+2+3 + #4 + #5 (this run): {final.macro_f1:.4f}")

    out = {
        "description": (
            "CIC IoT-DIAD 2024, calibrated externalized-Bayesian hybrid Gating "
            "Agent, D=10 -- fixes #1-#3 plus real calib_loss_delta (#4) and "
            "round-aware cosine calibration prompt (#5)."
        ),
        "delay": delay,
        "reference_rule_based_f1": 0.758,
        "reference_plain_llm_f1": 0.064,
        "reference_fixes123_f1": 0.2430,
        "final_macro_f1": final.macro_f1,
        "wall_clock_seconds": elapsed,
        "n_fallback_rounds": n_fallback_rounds,
        "n_rounds": len(sim.gating_history),
        "malicious_clients": malicious,
        "gating_history": sim.gating_history,
    }
    path = os.path.join(RESULTS_DIR, f"ciciot_diad_hybrid_d{delay}_fixes12345.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n-> {path}")


if __name__ == "__main__":
    main()
