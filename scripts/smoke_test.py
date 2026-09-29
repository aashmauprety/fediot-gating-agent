"""Fast correctness check: load a small slice of real N-BaIoT data, run a
few federated rounds with and without gating, and sanity-check that
things actually learn (macro-F1 should rise well above chance) before
committing to a longer overnight run. This is NOT a paper result -- it
uses tiny per-class sample counts and few rounds purely to validate the
code path.
"""
import sys
import time
sys.path.insert(0, "..")

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation

print("Loading a small N-BaIoT slice (n_max_per_class=300) ...")
t0 = time.time()
clients = load_nbaiot_federated(n_max_per_class=300, fine_grained=False, seed=0)
print(f"Loaded {len(clients)} clients in {time.time()-t0:.1f}s")
for cid, c in clients.items():
    print(f"  {cid}: {len(c.y_attack)} samples, classes present: {sorted(set(c.y_attack.tolist()))}")

print("\n=== No attack, no gating (sanity: should learn, F1 rising) ===")
sim = FederatedSimulation(
    clients, num_classes=3, embed_dim=115, malicious_clients=[],
    attack="none", gating=False, seed=0,
)
logs = sim.run(num_rounds=5)
for l in logs:
    print(f"  round {l.round_idx}: macro_f1={l.macro_f1:.3f}")
assert logs[-1].macro_f1 > 0.5, "sanity check failed: model did not learn on real data"
print("OK: model learns on real N-BaIoT data.")

print("\n=== Untargeted attack (alpha~33%), no gating vs gating ===")
malicious = list(clients.keys())[:3]
for gating in (False, True):
    sim = FederatedSimulation(
        clients, num_classes=3, embed_dim=115, malicious_clients=malicious,
        attack="untargeted", gating=gating, seed=0,
    )
    logs = sim.run(num_rounds=5)
    print(f"  gating={gating}: final macro_f1={logs[-1].macro_f1:.3f}")

print("\nSmoke test complete.")
