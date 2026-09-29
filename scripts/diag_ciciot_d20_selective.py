"""Diagnose the CIC IoT-DIAD trust_building D=20 selective-LLM failure
(F1=0.1869 vs the full-LLM reference of 0.898). Print every round's
per-client action, whether it went through the fast path or a real LLM
call, and the trust posterior mean/n_obs for the malicious clients, to
see exactly where this breaks down.
"""
import sys
sys.path.insert(0, "..")

from fedgate.data import load_ciciot_diad_attack_classification
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import SelectiveLLMGatingPolicy

MODEL = "qwen2.5:14b"


def main():
    clients = load_ciciot_diad_attack_classification(
        n_max_per_class=5000, n_clients=9, label_col="attack_group", seed=0,
    )
    classes = list(clients.values())[0].classes
    embed_dim = list(clients.values())[0].X.shape[1]
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    policy = SelectiveLLMGatingPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
    )
    sim = FederatedSimulation(
        clients=clients, num_classes=len(classes), embed_dim=embed_dim,
        malicious_clients=malicious, attack="trust_building", trust_building_delay=20,
        gating=True, seed=0, gating_policy=policy,
        trust_decay=0.9, force_accept_round0=True,
    )
    logs = sim.run(num_rounds=25, log_every=1)

    print("Per-round macro-F1:")
    for lg in logs:
        print(f"  round {lg.round_idx:2d}: F1={lg.macro_f1:.4f}")
    print(f"Final macro-F1 = {logs[-1].macro_f1:.4f}")
    print(f"LLM calls: {policy.n_llm_calls}/{policy.n_fast_path + policy.n_llm_calls}\n")
    for h in sim.gating_history:
        m_info = []
        for c in malicious:
            action = h["actions"][c]
            score = h["scores"][c]
            rationale = h["rationales"].get(c, "") if "rationales" in h else ""
            fast = "fast" if "fast path" in rationale else "LLM"
            mean = sim.trust.mean(c, "attack-head")
            n_obs = sim.trust.n_observations(c, "attack-head")
            m_info.append(f"{c}:{action}({score:.2f},{fast},post={mean:.2f}/n={n_obs:.1f})")
        h_info = []
        for c in client_ids:
            if c in malicious:
                continue
            action = h["actions"][c]
            h_info.append(f"{c}:{action}")
        print(f"round {h['round']:2d}: malicious=[{', '.join(m_info)}]")
        print(f"           honest=[{', '.join(h_info)}]")


if __name__ == "__main__":
    main()
