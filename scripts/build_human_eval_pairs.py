"""Human-evaluation pilot, step 2: from the paired source run
(human_eval_pilot_source_run.json -- real LLM rationale + shadow
rule-based rationale on the IDENTICAL digest every round), sample a
diverse, de-identified set of decision pairs for a blind comparison
survey.

Selection targets variety, not just "interesting" cases cherry-picked
to favor the LLM -- includes clean agreement cases (both exclude for
the same obvious reason) alongside the cases most likely to show a
real difference (trust-posterior override, honest-client borderline
calls) so the pilot isn't rigged by sample choice.
"""
import json
import os
import random

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
SRC_PATH = os.path.join(RESULTS_DIR, "human_eval_pilot_source_run.json")
OUT_PATH = os.path.join(RESULTS_DIR, "human_eval_pairs.json")

N_SAMPLE = 24
SEED = 0


def main():
    with open(SRC_PATH) as f:
        src = json.load(f)
    gh = src["gating_history"]
    malicious = set(src["malicious_clients"])

    candidates = []
    for h in gh:
        if h["round"] == 0:
            continue  # trivial force-accept, not a real judgment either way
        for cid, action in h["actions"].items():
            shadow_action = h["shadow_actions"][cid]
            candidates.append({
                "round": h["round"],
                "client_is_malicious": cid in malicious,
                "llm_action": action,
                "llm_rationale": h["rationales"][cid],
                "shadow_action": shadow_action,
                "shadow_rationale": h["shadow_rationales"][cid],
                "actions_agree": action == shadow_action,
                "has_trust_override": "posterior" in h["rationales"][cid] or "posterior" in h["shadow_rationales"][cid],
            })

    rng = random.Random(SEED)

    # Stratify: half from rounds where actions disagree (the most
    # interesting cases), half from rounds where they agree (so the
    # pilot isn't only showing cases engineered to favor the LLM).
    disagree = [c for c in candidates if not c["actions_agree"]]
    agree = [c for c in candidates if c["actions_agree"]]
    trust_override = [c for c in candidates if c["has_trust_override"]]

    rng.shuffle(disagree)
    rng.shuffle(agree)
    rng.shuffle(trust_override)

    n_disagree = min(len(disagree), N_SAMPLE // 3)
    n_override = min(len(trust_override), N_SAMPLE // 3)
    n_agree = N_SAMPLE - n_disagree - n_override

    selected = disagree[:n_disagree] + trust_override[:n_override] + agree[:n_agree]
    rng.shuffle(selected)

    pairs = []
    for i, c in enumerate(selected):
        # Randomize which side (A/B) the LLM rationale appears on, so a
        # rater can't learn "A is always the LLM" partway through.
        llm_first = rng.random() < 0.5
        pair = {
            "pair_id": i + 1,
            "round": c["round"],
            "action_A": c["llm_action"] if llm_first else c["shadow_action"],
            "explanation_A": c["llm_rationale"] if llm_first else c["shadow_rationale"],
            "action_B": c["shadow_action"] if llm_first else c["llm_action"],
            "explanation_B": c["shadow_rationale"] if llm_first else c["llm_rationale"],
            "_which_is_llm": "A" if llm_first else "B",  # answer key -- strip before sending to raters
            "_actions_agree": c["actions_agree"],
            "_has_trust_override": c["has_trust_override"],
        }
        pairs.append(pair)

    with open(OUT_PATH, "w") as f:
        json.dump({
            "description": (
                "Blind human-evaluation pilot pairs: LLM Gating Agent "
                "rationale vs. templated rule-based rationale, same "
                "underlying digest/decision event (CIC IoT-DIAD, D=10, "
                "qwen2.5:14b). _which_is_llm and other _-prefixed fields "
                "are the answer key -- strip them before handing pairs to "
                "raters."
            ),
            "n_pairs": len(pairs),
            "n_disagreeing_actions": n_disagree,
            "n_trust_override_cases": n_override,
            "n_agreeing_actions": n_agree,
            "pairs": pairs,
        }, f, indent=2)
    print(f"Wrote {len(pairs)} pairs -> {OUT_PATH}")
    print(f"  disagree={n_disagree}, trust_override={n_override}, agree={n_agree}")


if __name__ == "__main__":
    main()
