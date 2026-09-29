"""Colluding adaptive attacker (flagged as future work in the ICC paper's
Conclusion, now attempted): instead of each malicious client blending
toward its OWN clean-label gradient in isolation (Section VII-F,
`evasion_lambda`), a stronger, coordinated attacker pools all malicious
clients' clean gradients into ONE shared, less-noisy reference
(`evasion_collude=True`, `fedgate/experiment.py:_get_collude_reference`).
This requires malicious clients to share local data statistics with
each other -- a stronger threat-model assumption -- to test whether a
more representative reference evades detection better than each
client's own noisier local estimate did.

Tests against GShield (cheapest, already the "does this attack work at
all" reference) and the selective-LLM policy (the design this paper
proposes), targeted attack, CIC IoT-DIAD 2024, same lambda values used
throughout (0.0, 0.9, 0.99) for direct comparison.
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
OUT_PATH = os.path.join(RESULTS_DIR, "adaptive_evasion_colluding.json")
MODEL = "qwen2.5:14b"


def run_gshield(clients, classes, embed_dim, malicious, lam, num_rounds, seed):
    label = f"GShield, colluding, evasion_lambda={lam}"
    print(f"\n=== {label} ===", flush=True)
    sim = FederatedSimulation(
        clients, num_classes=len(classes), embed_dim=embed_dim,
        malicious_clients=malicious, attack="targeted",
        aggregation="gshield", seed=seed,
        scale_factor=1.0, evasion_lambda=lam, evasion_collude=True,
    )
    t0 = time.time()
    logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    elapsed = time.time() - t0
    final = logs[-1]
    print(f"F1={final.macro_f1:.4f}, targeted_success={final.targeted_success_rate} ({elapsed:.1f}s)", flush=True)
    return {
        "defense": "gshield", "evasion_lambda": lam,
        "final_macro_f1": final.macro_f1,
        "targeted_success_rate": final.targeted_success_rate,
        "wall_clock_seconds": elapsed,
    }


def run_selective_llm(clients, classes, embed_dim, malicious, lam, num_rounds, seed):
    label = f"Selective-LLM, colluding, evasion_lambda={lam}"
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
        scale_factor=1.0, evasion_lambda=lam, evasion_collude=True,
    )
    t0 = time.time()
    try:
        logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    except LLMGatingPolicyError as e:
        print(f"FAILED: {e}", flush=True)
        return {"defense": "selective_llm", "evasion_lambda": lam, "error": str(e)}
    elapsed = time.time() - t0
    final = logs[-1]
    malicious_fast = 0
    malicious_total = 0
    for h in sim.gating_history:
        for c in malicious:
            malicious_total += 1
            if "fast path" in h["rationales"].get(c, ""):
                malicious_fast += 1
    malicious_fast_frac = malicious_fast / malicious_total if malicious_total else 0.0
    print(f"F1={final.macro_f1:.4f}, targeted_success={final.targeted_success_rate}, "
          f"fast_path_bypass={malicious_fast}/{malicious_total} ({malicious_fast_frac:.1%}) ({elapsed:.1f}s)", flush=True)
    return {
        "defense": "selective_llm", "evasion_lambda": lam,
        "final_macro_f1": final.macro_f1,
        "targeted_success_rate": final.targeted_success_rate,
        "wall_clock_seconds": elapsed,
        "malicious_fast_path_bypass_fraction": malicious_fast_frac,
    }


def main(n_max_per_class=5000, n_clients=9, num_rounds=25, seed=0):
    print("Loading CIC IoT-DIAD 2024 attack data ...", flush=True)
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    lambdas = [0.0, 0.9, 0.99]
    results = []
    for lam in lambdas:
        results.append(run_gshield(clients, classes, embed_dim, malicious, lam, num_rounds, seed))
        with open(OUT_PATH, "w") as f:
            json.dump({"description": "Colluding adaptive attacker vs GShield and selective-LLM, targeted, CIC IoT-DIAD.", "runs": results}, f, indent=2)
    for lam in lambdas:
        results.append(run_selective_llm(clients, classes, embed_dim, malicious, lam, num_rounds, seed))
        with open(OUT_PATH, "w") as f:
            json.dump({"description": "Colluding adaptive attacker vs GShield and selective-LLM, targeted, CIC IoT-DIAD.", "runs": results}, f, indent=2)

    print("\n=== Summary ===", flush=True)
    for r in results:
        if "error" in r:
            print(f"  {r['defense']} lambda={r['evasion_lambda']}: FAILED")
        else:
            print(f"  {r['defense']} lambda={r['evasion_lambda']}: F1={r['final_macro_f1']:.4f}, "
                  f"targeted_success={r['targeted_success_rate']}")
    print(f"\n-> {OUT_PATH}")


if __name__ == "__main__":
    main()
