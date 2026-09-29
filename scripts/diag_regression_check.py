import sys
sys.path.insert(0, "..")
from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import SelectiveLLMGatingPolicy

clients = load_nbaiot_federated(n_max_per_class=3000, fine_grained=False, seed=0)
client_ids = list(clients.keys())
malicious = client_ids[:3]

policy = SelectiveLLMGatingPolicy(
    model="qwen2.5:14b", calibrated=True, warmup_rounds=5, warmup_scale=1.5,
    timeout_s=60.0, max_retries=3,
)
sim = FederatedSimulation(
    clients, num_classes=3, embed_dim=115,
    malicious_clients=malicious, attack="trust_building", trust_building_delay=10,
    gating=True, seed=0, gating_policy=policy, trust_decay=0.9, force_accept_round0=True,
)
logs = sim.run(num_rounds=25, log_every=25)
total = policy.n_fast_path + policy.n_llm_calls
print(f"F1= {logs[-1].macro_f1} (reference: 0.9971 pre-fix selective, 0.9974 full-LLM)", flush=True)
print(f"LLM calls: {policy.n_llm_calls}/{total} ({policy.n_llm_calls/total:.1%}) "
      f"(reference pre-fix: 44.9%)", flush=True)
