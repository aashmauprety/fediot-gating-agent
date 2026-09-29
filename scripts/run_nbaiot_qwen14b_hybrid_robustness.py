"""Continuation of the N-BaIoT generalization check (README "fourteenth
real result"): trust_building at D=0/10/20 all landed at F1=0.996-0.997
with qwen2.5:14b hybrid + full fix set. This script covers the two
attack types not yet tried on N-BaIoT with this configuration --
untargeted (random label-flipping, attacking from round 0) and targeted
(mislabels one specific class, attacking from round 0) -- matching the
CIC IoT-DIAD 2024 broad-validation sweep (README "twelfth real result")
which covered these same two attack types there.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import LLMExternalBayesianPolicy, LLMGatingPolicyError

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "nbaiot_qwen14b_hybrid_robustness.json")
MODEL = "qwen2.5:14b"


def run_one(clients, malicious, attack, num_rounds, seed):
    label = f"N-BaIoT, qwen2.5:14b hybrid + fixes, attack={attack}, seed={seed}"
    print(f"\n=== {label} ===", flush=True)
    policy = LLMExternalBayesianPolicy(
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
        print(f"FAILED: {e}", flush=True)
        return {"label": label, "attack": attack, "seed": seed, "error": str(e)}
    elapsed = time.time() - t0
    final = logs[-1]
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    client_ids = list(clients.keys())
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
          f"fallback {n_fallback_rounds}/{len(sim.gating_history)} rounds", flush=True)
    for h in sim.gating_history[-5:]:
        m_actions = [h["actions"][c] for c in malicious]
        h_actions = [h["actions"][c] for c in client_ids if c not in malicious]
        print(f"  round {h['round']:2d}: malicious={m_actions}  honest={h_actions}", flush=True)
    return {
        "label": label,
        "attack": attack,
        "seed": seed,
        "final_macro_f1": final.macro_f1,
        "targeted_success_rate": final.targeted_success_rate,
        "wall_clock_seconds": elapsed,
        "n_fallback_rounds": n_fallback_rounds,
        "n_rounds": len(sim.gating_history),
        "malicious_clients": malicious,
        "gating_history": sim.gating_history,
    }


def main(n_max_per_class=3000, num_rounds=25, seed=0):
    print(f"Loading N-BaIoT (n_max_per_class={n_max_per_class}) ...", flush=True)
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]
    print(f"{len(client_ids)} clients: {client_ids}", flush=True)

    results = []
    for attack in ["untargeted", "targeted"]:
        out = run_one(clients, malicious, attack, num_rounds, seed)
        results.append(out)
        with open(OUT_PATH, "w") as f:
            json.dump({
                "description": (
                    "N-BaIoT, qwen2.5:14b hybrid Gating Agent with the full "
                    "fix set -- untargeted/targeted attack-type robustness "
                    "sweep, completing the N-BaIoT generalization check "
                    "started with the trust_building D=0/10/20 sweep."
                ),
                "reference_trust_building": {"D=0": 0.9958, "D=10": 0.9974, "D=20": 0.9961},
                "runs": results,
            }, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for r in results:
        if "error" in r:
            print(f"  {r['label']}: FAILED", flush=True)
        else:
            print(f"  {r['label']}: F1={r['final_macro_f1']:.4f}, "
                  f"fallback {r['n_fallback_rounds']}/{r['n_rounds']}, "
                  f"targeted_success_rate={r['targeted_success_rate']}", flush=True)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
