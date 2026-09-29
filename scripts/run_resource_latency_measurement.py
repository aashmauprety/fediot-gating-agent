"""Real per-decision latency/CPU/memory measurement for the selective-LLM
Gating Agent, styled after (not copying the specific numbers of) the
resource-analysis approach in "Think Fast: Real-Time IoT Intrusion
Reasoning Using IDS and LLMs at the Edge Gateway" (Jamshidi et al.,
IEEE IoT Journal, 2026) -- box-plot-style latency/CPU/memory per
decision, plus a statistical test on whether the fast-path/LLM-call
split is meaningfully different (it should be, by construction: this
IS the mechanism the cost-reduction paper section claims).

Honesty note: we do NOT report energy consumption (Joules) -- Think
Fast used real wattmeter instrumentation on Raspberry Pi hardware; we
have no such instrumentation on this dev machine, and estimating it
would be fabrication. Latency, CPU%, and memory are directly measured
here via wall-clock timing and `psutil` sampling of the Ollama server
process (where the actual LLM inference happens), not simulated.
"""
import json
import os
import sys
import time

sys.path.insert(0, "..")

import psutil

from fedgate.data import load_nbaiot_federated
from fedgate.experiment import FederatedSimulation
from fedgate.gating_agent import SelectiveLLMGatingPolicy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "resource_latency_measurement.json")
MODEL = "qwen2.5:14b"


def find_ollama_process():
    """The actual model inference happens in a child `llama-server` runner
    process spawned by `ollama serve`, not in `ollama serve` itself (which
    is just a lightweight coordinator) -- sampling the parent alone
    drastically undercounts CPU/memory (confirmed via `ps aux`: the runner
    holds ~10GB RSS for qwen2.5:14b, `ollama serve` itself under 40MB)."""
    for p in psutil.process_iter(["pid", "name", "cmdline"]):
        try:
            cmdline = " ".join(p.info.get("cmdline") or [])
            if "llama-server" in cmdline:
                return psutil.Process(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return None


class InstrumentedSelectiveLLMGatingPolicy(SelectiveLLMGatingPolicy):
    def __init__(self, *args, ollama_proc=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ollama_proc = ollama_proc
        self.records = []

    def decide(self, *args, **kwargs):
        fast_before = self.n_fast_path
        cpu_t0 = None
        if self.ollama_proc is not None:
            try:
                cpu_t0 = self.ollama_proc.cpu_times()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        t0 = time.time()
        result = super().decide(*args, **kwargs)
        dt = time.time() - t0
        is_fast = self.n_fast_path > fast_before
        cpu_pct = None
        mem_mb = None
        if self.ollama_proc is not None:
            try:
                # Call-local CPU%: (user+sys CPU-seconds consumed DURING
                # this call) / (wall-clock seconds this call took) * 100 --
                # not `cpu_percent(interval=None)`, which measures usage
                # since the LAST sample and would be diluted by unrelated
                # client-training work happening between gating decisions.
                cpu_t1 = self.ollama_proc.cpu_times()
                if cpu_t0 is not None and dt > 0:
                    cpu_delta = (cpu_t1.user - cpu_t0.user) + (cpu_t1.system - cpu_t0.system)
                    cpu_pct = 100.0 * cpu_delta / dt
                mem_mb = self.ollama_proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        self.records.append({
            "latency_s": dt, "fast_path": is_fast,
            "cpu_percent": cpu_pct, "memory_mb": mem_mb,
        })
        return result


def main(n_max_per_class=3000, num_rounds=25, seed=0):
    # Warm up so the llama-server runner process actually exists before we
    # try to find and sample it (it's spawned on first model use, not at
    # `ollama serve` startup).
    import urllib.request as _ur
    import json as _json
    try:
        req = _ur.Request(
            "http://localhost:11434/api/generate",
            data=_json.dumps({"model": MODEL, "prompt": "hi", "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        _ur.urlopen(req, timeout=60).read()
    except Exception as e:
        print(f"Warm-up call failed (continuing anyway): {e}", flush=True)

    ollama_proc = find_ollama_process()
    print(f"Ollama runner process: {ollama_proc}", flush=True)
    if ollama_proc is not None:
        ollama_proc.cpu_percent(interval=None)  # prime the counter

    print("Loading N-BaIoT ...", flush=True)
    clients = load_nbaiot_federated(n_max_per_class=n_max_per_class, fine_grained=False, seed=0)
    client_ids = list(clients.keys())
    malicious = client_ids[:3]

    policy = InstrumentedSelectiveLLMGatingPolicy(
        model=MODEL, calibrated=True, warmup_rounds=5, warmup_scale=1.5,
        timeout_s=60.0, max_retries=3, ollama_proc=ollama_proc,
    )
    sim = FederatedSimulation(
        clients, num_classes=3, embed_dim=115,
        malicious_clients=malicious, attack="trust_building", trust_building_delay=10,
        gating=True, seed=seed, gating_policy=policy,
        trust_decay=0.9, force_accept_round0=True,
    )
    t0 = time.time()
    logs = sim.run(num_rounds=num_rounds, log_every=num_rounds)
    elapsed = time.time() - t0
    print(f"Run done: F1={logs[-1].macro_f1:.4f} ({elapsed:.1f}s), "
          f"{len(policy.records)} decisions recorded", flush=True)

    fast = [r for r in policy.records if r["fast_path"]]
    llm = [r for r in policy.records if not r["fast_path"]]

    def stats(rows, key):
        vals = [r[key] for r in rows if r[key] is not None]
        if not vals:
            return None
        import statistics as st
        return {
            "n": len(vals), "mean": st.mean(vals),
            "std": st.stdev(vals) if len(vals) > 1 else 0.0,
            "min": min(vals), "max": max(vals),
        }

    summary = {
        "fast_path": {
            "n": len(fast),
            "latency_s": stats(fast, "latency_s"),
            "cpu_percent": stats(fast, "cpu_percent"),
            "memory_mb": stats(fast, "memory_mb"),
        },
        "llm_call": {
            "n": len(llm),
            "latency_s": stats(llm, "latency_s"),
            "cpu_percent": stats(llm, "cpu_percent"),
            "memory_mb": stats(llm, "memory_mb"),
        },
    }

    # Welch's t-test on latency (fast-path vs LLM-call) -- unequal variances,
    # matching the ANOVA/Tukey-style rigor the Think Fast comparison paper uses.
    try:
        from scipy import stats as scipy_stats
        fast_lat = [r["latency_s"] for r in fast]
        llm_lat = [r["latency_s"] for r in llm]
        if len(fast_lat) > 1 and len(llm_lat) > 1:
            t_stat, p_val = scipy_stats.ttest_ind(llm_lat, fast_lat, equal_var=False)
            summary["welch_ttest_latency"] = {"t_stat": float(t_stat), "p_value": float(p_val)}
    except ImportError:
        pass

    print(json.dumps(summary, indent=2), flush=True)
    with open(OUT_PATH, "w") as f:
        json.dump({
            "description": (
                "Real per-decision latency/CPU/memory for the selective-LLM "
                "Gating Agent, fast-path vs real LLM call, N-BaIoT TB D=10. "
                "No energy (Joules) measurement -- no wattmeter on this "
                "machine; would be fabrication to estimate it."
            ),
            "summary": summary,
            "raw_records": policy.records,
        }, f, indent=2)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
