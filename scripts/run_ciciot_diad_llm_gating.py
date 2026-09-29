"""First real test of the LLM Gating Agent on CIC IoT-DIAD 2024's richer
feature set (Section on fixing the LLM Gating Agent weakness). On
N-BaIoT (115 flow-statistic features), five real LLM configurations
(single-turn 8B, +FetchHistory, 14B, externalized-Bayesian hybrid,
hybrid+calibration) all failed to beat the rule-based policy at the
D=10 trust-building delay (rule-based: 0.986; best LLM attempt: 0.386).
The hypothesis here: does a richer underlying feature set (118 features
vs. 115, including protocol/timing/header signal) change the
*classifier's* separability enough that the digest an LLM sees (still
just 3 scalars: update-norm z-score, cosine similarity, calibration-loss
delta) becomes more informative, i.e. honest vs. malicious clients'
digests look more different -- even though the digest's own dimensionality
is unchanged?

Real rule-based reference just established
(run_ciciot_diad_trustbuild.py): D=0 gating_with_bayesian=0.098,
D=10=0.758, D=20=0.367. Same protocol here, single-turn Llama 3.1 8B
(the fastest, first-tried LLM config on N-BaIoT), D in {0, 10}.
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
MODEL = "llama3.1:8b"


def main(n_max_per_class=5000, n_clients=9, num_rounds=25):
    print(f"Loading CIC IoT-DIAD 2024 attack data ...")
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=n_max_per_class, n_clients=n_clients, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    print(f"{len(classes)} classes, embed_dim={embed_dim}, {len(clients)} clients")

    client_ids = list(clients.keys())
    malicious = client_ids[:3]
    delays = [0, 10]
    results = []
    for delay in delays:
        print(f"\n=== delay={delay} (model={MODEL}, single-turn LLM gating) ===")
        policy = LLMGatingPolicy(model=MODEL)
        sim = FederatedSimulation(
            clients, num_classes=len(classes), embed_dim=embed_dim,
            malicious_clients=malicious, attack="trust_building",
            gating=True, trust_building_delay=delay, seed=0,
            gating_policy=policy,
        )
        t0 = time.time()
        try:
            logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
        except LLMGatingPolicyError as e:
            print(f"  FAILED at delay={delay}: {e}")
            results.append({"delay_rounds": delay, "strategy": f"gating_llm_{MODEL}", "error": str(e)})
            continue
        elapsed = time.time() - t0
        final = logs[-1]
        row = {
            "delay_rounds": delay,
            "strategy": f"gating_llm_{MODEL}",
            "final_macro_f1": final.macro_f1,
            "wall_clock_seconds": elapsed,
            "sample_actions": final.actions,
            "sample_gating_scores": final.gating_scores,
        }
        results.append(row)
        print(f"  delay={delay:3d} F1={final.macro_f1:.3f} ({elapsed:.1f}s)")
        path = os.path.join(RESULTS_DIR, "ciciot_diad_trustbuild_llm.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  -> {path} (updated)")

    print(f"\n=== Summary vs. rule-based reference (D=0: 0.098, D=10: 0.758) ===")
    for r in results:
        if "final_macro_f1" in r:
            print(f"  D={r['delay_rounds']}: LLM={r['final_macro_f1']:.3f}")


if __name__ == "__main__":
    main()
