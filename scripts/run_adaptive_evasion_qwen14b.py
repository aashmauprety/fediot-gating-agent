"""Phase 2 of the adaptive-attacker experiment (see
run_adaptive_evasion_gshield_sweep.py for Phase 1 and full context):
test the qwen2.5:14b hybrid Gating Agent under the same
constrain-and-scale-style adaptive attack (each malicious client blends
its poisoned update toward its OWN clean-label gradient,
`evasion_lambda` in [0,1]) that GShield-style proved robust to in
aggregate F1 terms (0.917-0.923 across the whole lambda range), despite
a real, confirmed per-round mechanism where 2 of 3 malicious clients
slip into the "benign" cluster at high lambda.

Tests lambda in {0.0, 0.9, 0.99} -- the baseline, and the two values
where GShield's per-round clustering was directly confirmed to
misclassify malicious clients as benign -- on the targeted attack (has
a clean, direct damage metric: the targeted-class misclassification
rate), to see whether the LLM Gating Agent's richer per-client digest
(update norm z-score, cosine z-score, AND calibration-loss direction,
not just a PCA-projected clustering signal) still catches what GShield's
clustering missed, or shows the same flat robustness.
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
OUT_PATH = os.path.join(RESULTS_DIR, "adaptive_evasion_qwen14b.json")
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
        label = f"targeted, evasion_lambda={lam}, qwen2.5:14b hybrid"
        print(f"\n=== {label} ===", flush=True)
        policy = LLMExternalBayesianPolicy(
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
        print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
              f"fallback {n_fallback_rounds}/{len(sim.gating_history)}, "
              f"targeted_success={final.targeted_success_rate}", flush=True)
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
            "malicious_clients": malicious,
            "gating_history": sim.gating_history,
        })
        with open(OUT_PATH, "w") as f:
            json.dump({
                "description": (
                    "Qwen2.5 14B hybrid Gating Agent under the constrain-and-"
                    "scale adaptive attack (each malicious client blends toward "
                    "its own clean-label gradient), targeted attack, CIC "
                    "IoT-DIAD 2024. Compare against GShield-style baseline at "
                    "the same lambda values in adaptive_evasion_gshield_sweep.json "
                    "(0.9209/0.9208/0.9207 macro-F1, targeted_success ~1%, flat "
                    "across all lambda)."
                ),
                "runs": results,
            }, f, indent=2)

    print("\n=== Summary ===", flush=True)
    print("GShield-style reference (same lambdas): F1=0.9209/0.9208/0.9207, targeted_success~1%")
    for r in results:
        if "error" in r:
            print(f"  lambda={r['evasion_lambda']}: FAILED")
        else:
            print(f"  lambda={r['evasion_lambda']}: F1={r['final_macro_f1']:.4f}, "
                  f"targeted_success={r['targeted_success_rate']}, "
                  f"fallback={r['n_fallback_rounds']}/{r['n_rounds']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
