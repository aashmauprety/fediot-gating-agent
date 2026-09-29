"""Diagnose WHY adding the FetchHistory tool option made Llama 3.1 8B's
discrimination collapse (README 'second real result': every client got
`downweight` with a score clustered at 0.3-0.4 regardless of whether its
digest looked normal or anomalous, once the tool option was added).

Three prompt variants tested on the SAME 5 synthetic cases, isolating
two candidate causes:
  A) SYSTEM_PROMPT_WITH_TOOL as-is (tool option present, model may call it)
  B) SYSTEM_PROMPT_WITH_TOOL text MINUS the fetch_history sentence (same
     added trust-building framing/background, but no tool option at all)
  C) SYSTEM_PROMPT_NO_TOOL (original control, known to discriminate fine)

If B collapses the same way A does, the cause is the added prompt text/
framing itself (distraction), not the two-turn tool-calling mechanism.
If B discriminates fine (like C) and only A collapses, the cause is
specific to the tool-calling mechanism (e.g. the model second-guessing
itself into a hedge once a fetch_history option exists at all, even
when it doesn't take it).
"""
import json
import os
import sys
import urllib.request

sys.path.insert(0, "..")

from fedgate.gating_agent import BayesianTrust, Digest, LLMGatingPolicy

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
OUT_PATH = os.path.join(RESULTS_DIR, "diag_fetchhistory_regression.json")
MODEL = "llama3.1:8b"

# Variant B: same background/framing as WITH_TOOL, but the tool-offering
# sentence is removed and replaced with a direct instruction to always
# decide now -- isolates the added prose from the tool mechanism itself.
SYSTEM_PROMPT_FRAMING_ONLY_NO_TOOL = LLMGatingPolicy.SYSTEM_PROMPT_WITH_TOOL.replace(
    'If the posterior alone is not enough to judge '
    "whether this round's anomaly is a one-off against a long "
    "history of good behavior or part of a real pattern, you may "
    'respond with EXACTLY {"action": "fetch_history"} to request '
    "that gateway's actual recent per-round decision record before "
    "deciding -- do this when the gateway has a high posterior mean "
    "but this round looks anomalous, since that is exactly the "
    "ambiguous case history resolves. Otherwise, or after you "
    "receive the history, respond",
    "Always decide now, directly, using only the posterior mean and "
    "this round's digest -- there is no way to fetch additional history. "
    "Respond",
)
assert SYSTEM_PROMPT_FRAMING_ONLY_NO_TOOL != LLMGatingPolicy.SYSTEM_PROMPT_WITH_TOOL, \
    "replacement did not match -- prompt text drifted, fix the string above"


def call_ollama(system_prompt, user_prompt, model=MODEL, timeout=60):
    payload = json.dumps({
        "model": model,
        "system": system_prompt,
        "prompt": user_prompt,
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0},
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:11434/api/generate", data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    return body.get("response", "")


def build_user_prompt(digest, z, cos_z, calib_words, posterior_mean, n_obs):
    return (
        f"This round's digest for gateway {digest.client_id}:\n"
        f"  update norm z-score vs. other gateways this round: {z:+.2f}\n"
        f"  cosine similarity z-score vs. those same gateways this round: {cos_z:+.2f}\n"
        f"  calibration loss: {calib_words}\n"
        f"Long-term trust posterior: mean={posterior_mean:.2f}, observed over {n_obs:.0f} rounds.\n"
        "How should this update be gated?"
    )


CASES = [
    dict(name="clearly_normal", z=0.1, cos_z=0.1, calib="essentially unchanged",
         posterior_mean=0.85, n_obs=15),
    dict(name="clearly_anomalous", z=3.8, cos_z=-3.5, calib="got much worse",
         posterior_mean=0.5, n_obs=3),
    dict(name="ambiguous_strong_history", z=2.5, cos_z=-2.0, calib="got slightly worse",
         posterior_mean=0.95, n_obs=20),
    dict(name="ambiguous_weak_history", z=2.5, cos_z=-2.0, calib="got slightly worse",
         posterior_mean=0.6, n_obs=2),
    dict(name="borderline", z=1.6, cos_z=-1.4, calib="got slightly worse",
         posterior_mean=0.7, n_obs=8),
]

VARIANTS = {
    "A_with_tool": LLMGatingPolicy.SYSTEM_PROMPT_WITH_TOOL,
    "B_framing_only_no_tool": SYSTEM_PROMPT_FRAMING_ONLY_NO_TOOL,
    "C_original_no_tool": LLMGatingPolicy.SYSTEM_PROMPT_NO_TOOL,
}


def main():
    results = {}
    for variant_name, system_prompt in VARIANTS.items():
        print(f"\n=== Variant {variant_name} ===", flush=True)
        variant_results = []
        for case in CASES:
            user_prompt = build_user_prompt(
                Digest(client_id="gw_test", component="attack-head", update_norm=0, cosine_sim_to_prev_update=0, calib_loss_delta=0),
                case["z"], case["cos_z"], case["calib"], case["posterior_mean"], case["n_obs"],
            )
            raw = call_ollama(system_prompt, user_prompt)
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = {"_unparseable": raw}
            print(f"  {case['name']:28s} -> {parsed}", flush=True)
            variant_results.append({"case": case["name"], "response": parsed})
        results[variant_name] = variant_results

    with open(OUT_PATH, "w") as f:
        json.dump({
            "description": (
                "Diagnosing the FetchHistory-tool regression (Llama 3.1 8B): "
                "does the collapse come from the added prompt framing/text, "
                "or from the tool-calling mechanism itself? Variant B isolates "
                "the framing by keeping the same added background text but "
                "removing the fetch_history option entirely."
            ),
            "model": MODEL,
            "cases": CASES,
            "results": results,
        }, f, indent=2)
    print(f"\n-> {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
