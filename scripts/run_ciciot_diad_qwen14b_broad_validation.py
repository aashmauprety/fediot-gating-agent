"""Broaden the "eleventh real result" (README) before trusting it as a
paper number: qwen2.5:14b + the z-scored-cosine/worded-calib-delta hybrid
Gating Agent hit F1=0.9041 on CIC IoT-DIAD 2024, D=10, trust_building
attack, seed=0 -- but that is one seed, one delay, one attack type.

This script repeats the identical policy/fix configuration
(LLMExternalBayesianPolicy(model="qwen2.5:14b", calibrated=True,
warmup_rounds=5, warmup_scale=1.5), FederatedSimulation(trust_decay=0.9,
force_accept_round0=True)) across:

  - trust_building, D=10, seeds 1 and 2 (does the D=10/seed=0 result
    replicate, or was it a lucky seed?)
  - trust_building, D=20 (attacker waits twice as long before starting --
    does the fix still work with a longer honest history to poison?)
  - untargeted attack (malicious clients attack from round 0, no
    trust-building delay at all -- a different attack shape entirely)
  - targeted attack (same, but mislabels one specific class instead of
    random label-flipping)

Each run is ~10-15 minutes with this model, so this is a long sequential
job (roughly 75-90 minutes total) -- results are written incrementally
to the output JSON after each run finishes, not only at the very end, so
a partial run is still useful if interrupted.
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


def run_one(clients, classes, embed_dim, malicious, attack, delay, num_rounds, seed, label):
    print(f"\n=== {label} ===", flush=True)
    policy = LLMExternalBayesianPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
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
        return {"label": label, "attack": attack, "delay": delay, "seed": seed, "error": str(e)}
    elapsed = time.time() - t0
    final = logs[-1]
    n_fallback_rounds = sum(1 for h in sim.gating_history if h["fallback_triggered"])
    client_ids = list(clients.keys())
    print(f"Final macro-F1 = {final.macro_f1:.4f} ({elapsed:.1f}s), "
          f"fallback {n_fallback_rounds}/{len(sim.gating_history)} rounds", flush=True)
    last_few = sim.gating_history[-5:]
    for h in last_few:
        m_actions = [h["actions"][c] for c in malicious]
        h_actions = [h["actions"][c] for c in client_ids if c not in malicious]
        print(f"  round {h['round']:2d}: malicious={m_actions}  honest={h_actions}", flush=True)
    return {
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


def main(n_max_per_class=5000, n_clients=9, num_rounds=25):
    print("Loading CIC IoT-DIAD 2024 attack data ...", flush=True)
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    print(f"{len(classes)} classes, embed_dim={embed_dim}, {len(clients)} clients", flush=True)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    runs_spec = [
        ("trust_building", 10, 1, "trust_building D=10 seed=1 (replicate)"),
        ("trust_building", 10, 2, "trust_building D=10 seed=2 (replicate)"),
        ("trust_building", 20, 0, "trust_building D=20 seed=0 (longer honest history)"),
        ("untargeted", 0, 0, "untargeted (attacks from round 0) seed=0"),
        ("targeted", 0, 0, "targeted (attacks from round 0) seed=0"),
    ]

    results = []
    for attack, delay, seed, label in runs_spec:
        out = run_one(clients, classes, embed_dim, malicious, attack, delay, num_rounds, seed, label)
        results.append(out)
        # write incrementally so a partial run is still useful
        with open(OUT_PATH, "w") as f:
            json.dump({
                "description": (
                    "Broader validation of the qwen2.5:14b hybrid Gating Agent "
                    "fix (README 'eleventh real result', originally F1=0.9041 "
                    "on trust_building D=10 seed=0 only)."
                ),
                "reference_d10_seed0_f1": 0.9041,
                "reference_rule_based_d10_f1": 0.758,
                "runs": results,
            }, f, indent=2)

    print("\n=== Summary ===", flush=True)
    print("Reference: trust_building D=10 seed=0 (original result) = 0.9041")
    for r in results:
        if "error" in r:
            print(f"  {r['label']}: FAILED ({r['error']})", flush=True)
        else:
            print(f"  {r['label']}: F1={r['final_macro_f1']:.4f}, "
                  f"fallback {r['n_fallback_rounds']}/{r['n_rounds']}", flush=True)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
