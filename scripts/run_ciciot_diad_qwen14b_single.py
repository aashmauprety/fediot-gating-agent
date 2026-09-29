"""Run a single config of the broad-validation sweep
(run_ciciot_diad_qwen14b_broad_validation.py) -- split out because the
combined sweep hit repeated Ollama timeouts partway through (server
degraded under sustained memory pressure after ~15-20 min of continuous
inference; a clean `ollama serve` restart fixed it immediately, trivial
calls went from ~22s back to ~2s). Re-running one config at a time with
a restart in between, and a longer per-call timeout for resilience,
avoids depending on one uninterrupted ~90-minute process.

Usage: python3 run_ciciot_diad_qwen14b_single.py <attack> <delay> <seed> <label>
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
OUT_PATH = os.path.join(RESULTS_DIR, "ciciot_diad_qwen14b_broad_validation.json")
MODEL = "qwen2.5:14b"


def main(attack, delay, seed, label, n_max_per_class=5000, n_clients=9, num_rounds=25):
    print("Loading CIC IoT-DIAD 2024 attack data ...", flush=True)
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    print(f"\n=== {label} ===", flush=True)
    policy = LLMExternalBayesianPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
        timeout_s=60.0, max_retries=3,
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
        result = {"label": label, "attack": attack, "delay": delay, "seed": seed, "error": str(e)}
    else:
        elapsed = time.time() - t0
        final = logs[-1]
        n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
        print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
              f"fallback {n_fallback_rounds}/{len(sim.gating_history)} rounds", flush=True)
        for h in sim.gating_history[-5:]:
            m_actions = [h["actions"][c] for c in malicious]
            h_actions = [h["actions"][c] for c in client_ids if c not in malicious]
            print(f"  round {h['round']:2d}: malicious={m_actions}  honest={h_actions}", flush=True)
        result = {
            "label": label,
            "attack": attack,
            "delay": delay,
            "seed": seed,
            "final_macro_f1": final.macro_f1,
            "targeted_success_rate": final.targeted_success_rate,
            "wall_clock_seconds": elapsed,
            "n_fallback_rounds": n_fallback_rounds,
            "n_rounds": len(sim.gating_history),
            "malicious_clients": malicious,
            "gating_history": sim.gating_history,
        }

    existing = {"runs": []}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            existing = json.load(f)
    # replace any prior entry with the same label, else append
    existing["runs"] = [r for r in existing.get("runs", []) if r.get("label") != label]
    existing["runs"].append(result)
    existing["description"] = (
        "Broader validation of the qwen2.5:14b hybrid Gating Agent fix "
        "(README 'eleventh real result', originally F1=0.9041 on "
        "trust_building D=10 seed=0 only)."
    )
    existing["reference_d10_seed0_f1"] = 0.9041
    existing["reference_rule_based_d10_f1"] = 0.758
    with open(OUT_PATH, "w") as f:
        json.dump(existing, f, indent=2)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    attack, delay, seed, label = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    main(attack, delay, seed, label)
