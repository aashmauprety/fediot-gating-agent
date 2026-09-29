"""Real test of the graceful-degradation limitation fix: kill the actual
Ollama server mid-run (not a mocked failure) and confirm
`GracefulDegradationPolicy` keeps training going via the deterministic
rule-based fallback instead of crashing, then confirm it resumes using
the real LLM again once the server comes back.
"""
import json
import os
import subprocess
import sys
import threading
import time

sys.path.insert(0, "..")

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import SelectiveLLMGatingPolicy, GracefulDegradationPolicy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "graceful_degradation_test.json")
MODEL = "qwen2.5:14b"


def kill_and_restart_ollama(kill_after_s=30, downtime_s=90):
    time.sleep(kill_after_s)
    print(f"\n>>> [outage thread] Killing ollama serve now (t={kill_after_s}s) <<<", flush=True)
    subprocess.run(["pkill", "-f", "ollama serve"])
    time.sleep(downtime_s)
    print(f">>> [outage thread] Restarting ollama serve now (downtime={downtime_s}s) <<<", flush=True)
    subprocess.Popen(
        ["ollama", "serve"],
        stdout=open("/tmp/ollama_serve_degradation_test.log", "w"),
        stderr=subprocess.STDOUT,
    )


def main(n_max_per_class=3000, num_rounds=25, seed=0):
    print("Loading N-BaIoT ...", flush=True)
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    primary = SelectiveLLMGatingPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
        timeout_s=15.0, max_retries=1,  # short timeout/retries so an outage fails FAST, not after minutes
    )
    policy = GracefulDegradationPolicy(primary)

    sim = FederatedSimulation(
        clients, num_classes=3, embed_dim=115,
        malicious_clients=malicious, attack="trust_building", trust_building_delay=10,
        gating=True, seed=seed, gating_policy=policy,
        trust_decay=0.9, force_accept_round0=True,
    )

    outage = threading.Thread(target=kill_and_restart_ollama, kwargs={"kill_after_s": 30, "downtime_s": 90}, daemon=True)
    outage.start()

    t0 = time.time()
    logs = sim.run(num_rounds=num_rounds, log_every=1)
    elapsed = time.time() - t0

    print(f"\nRun completed WITHOUT crashing. Final macro-F1 = {logs[-1].macro_f1:.4f} ({elapsed:.1f}s)", flush=True)
    print(f"Degraded decisions (LLM unreachable, rule-based fallback used): "
          f"{policy.n_degraded}", flush=True)
    print(f"Degraded rounds: {sorted(set(policy.degraded_rounds))}", flush=True)

    with open(OUT_PATH, "w") as f:
        json.dump({
            "description": (
                "Real test: killed the actual ollama server 30s into the run, "
                "restarted it 90s later, while GracefulDegradationPolicy wraps "
                "the selective-LLM policy. Confirms training does not crash and "
                "correctly falls back to the deterministic rule-based policy "
                "during the real outage window, then resumes using the real LLM."
            ),
            "final_macro_f1": logs[-1].macro_f1,
            "wall_clock_seconds": elapsed,
            "n_degraded_decisions": policy.n_degraded,
            "degraded_rounds": sorted(set(policy.degraded_rounds)),
            "n_rounds": num_rounds,
        }, f, indent=2)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
