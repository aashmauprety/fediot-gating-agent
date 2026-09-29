"""Zero-shot device onboarding (Section VI-B, "Zero-Shot Device
Onboarding"): kNN retrieval over a device-fingerprint knowledge base built
from admitted device embeddings, then a confidence-based
admit/defer/escalate decision.

`RuleBasedOnboardingPolicy` is the documented fixed-threshold stand-in
used for most of this paper's onboarding results. `LLMOnboardingPolicy`
(below) is a real implementation, backed by a local Ollama server, the
same pattern that was diagnosed and fixed for the Gating Agent
(gating_agent.py): it is given the k-NN neighbor list as an
ALREADY-NORMALIZED vote-share distribution (not raw, scale-dependent
embedding distances -- the same lesson learned from the Gating Agent's
digest design), plus calibration few-shot examples spanning
clear-match/ambiguous/no-match, and produces admit/defer/escalate with a
real natural-language rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np

OnboardAction = Literal["auto-admit", "request-more-data", "escalate"]


@dataclass
class FingerprintEntry:
    label: str
    embedding: np.ndarray


class DeviceFingerprintKB:
    def __init__(self):
        self.entries: List[FingerprintEntry] = []

    def add(self, label: str, embedding: np.ndarray) -> None:
        self.entries.append(FingerprintEntry(label, embedding))

    def knn(self, query: np.ndarray, k: int = 5) -> List[Tuple[FingerprintEntry, float]]:
        if not self.entries:
            return []
        dists = [
            (e, float(np.linalg.norm(e.embedding - query))) for e in self.entries
        ]
        dists.sort(key=lambda t: t[1])
        return dists[:k]


class RuleBasedOnboardingPolicy:
    """Stand-in for the LLM Gating Agent's onboarding reasoning. Confidence
    is a simple inverse-distance vote among the k nearest fingerprints;
    thresholds are documented, not learned."""

    def __init__(self, k: int = 5, admit_thresh: float = 0.7, escalate_thresh: float = 0.35):
        self.k = k
        self.admit_thresh = admit_thresh
        self.escalate_thresh = escalate_thresh

    def decide(
        self, kb: DeviceFingerprintKB, query_embedding: np.ndarray
    ) -> Tuple[OnboardAction, str, float, str]:
        neighbors = kb.knn(query_embedding, self.k)
        if not neighbors:
            return "escalate", "no fingerprints in knowledge base yet", 0.0, ""

        weights = {}
        for entry, dist in neighbors:
            w = 1.0 / (dist + 1e-6)
            weights[entry.label] = weights.get(entry.label, 0.0) + w
        total = sum(weights.values())
        best_label = max(weights, key=weights.get)
        confidence = weights[best_label] / total

        rationale = (
            f"{self.k}-NN vote over device-fingerprint KB: top label '{best_label}' "
            f"received {confidence:.0%} of inverse-distance weight among neighbors "
            f"{[ (e.label, round(d,3)) for e, d in neighbors]}."
        )
        if confidence >= self.admit_thresh:
            return "auto-admit", rationale, confidence, best_label
        elif confidence >= self.escalate_thresh:
            return "request-more-data", rationale, confidence, best_label
        else:
            return "escalate", rationale, confidence, best_label


class LLMOnboardingPolicyError(RuntimeError):
    """Raised on any failure to get a valid decision from the LLM backend.
    Never silently downgraded to RuleBasedOnboardingPolicy -- a caller
    that wants a fallback must do so explicitly."""


class LLMOnboardingPolicy:
    """Real LLM-based onboarding agent, backed by a local Ollama server
    (https://ollama.com). Given the SAME k-NN retrieval this module
    already does, it reasons over the normalized vote-share distribution
    (not raw embedding distances, which are not meaningfully comparable
    across queries/tasks the way a same-round cross-client z-score was
    for the Gating Agent) to decide auto-admit / request-more-data /
    escalate, with a real natural-language rationale."""

    SYSTEM_PROMPT = (
        "You are a device-onboarding agent for a federated IoT device-"
        "identification system. A knowledge base holds a fingerprint "
        "(representative traffic embedding) for each already-admitted "
        "device type. When traffic from a device arrives, you are given "
        "its k nearest fingerprint neighbors as a vote-share distribution "
        "(each neighbor's share of an inverse-distance-weighted vote, "
        "summing to 1.0 across all k neighbors) -- a normalized measure "
        "of how much closer this query is to each candidate device type "
        "than to the others, not a raw embedding distance. Decide: "
        '"auto-admit" (confidently label this traffic as one specific '
        'already-known device type), "request-more-data" (plausibly a '
        "known type, one candidate leads but not decisively, worth more "
        'evidence before committing), or "escalate" (no candidate wins '
        "a meaningfully larger share than the others -- the query does "
        "not resemble any single known device type well enough, likely "
        "a genuinely new device type).\n\n"
        "Calibration reference:\n"
        "Example 1 (clear match): top candidate's vote share = 0.85, "
        "next-highest = 0.08 -> auto-admit, predicted_label = top "
        "candidate (one candidate dominates overwhelmingly).\n"
        "Example 2 (ambiguous): top candidate's vote share = 0.45, "
        "next-highest = 0.30 -> request-more-data, predicted_label = top "
        "candidate (a plausible lead, but two candidates are close "
        "enough that more evidence would change the answer).\n"
        "Example 3 (no good match): top candidate's vote share = 0.22, "
        "next-highest = 0.19, spread across many candidates -> escalate, "
        "predicted_label = top candidate anyway, for the record, but "
        "flagged as a probably-new device type since nothing stands out.\n\n"
        "Now judge the real query below the same way. Respond with ONLY "
        'a JSON object of the exact form {"action": "auto-admit" | '
        '"request-more-data" | "escalate", "predicted_label": "<the top '
        'candidate label, always>", "confidence": <number between 0.0 and '
        '1.0, your own judgment of how confident this match is>, '
        '"rationale": "<one sentence, naming the specific vote-share '
        'values that drove the decision>"}. No text before or after the '
        "JSON."
    )

    def __init__(
        self,
        model: str = "qwen2.5:14b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 60.0,
        max_retries: int = 3,
        k: int = 5,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.k = k

    def _build_prompt(self, weights: dict, neighbors) -> str:
        total = sum(weights.values())
        shares = sorted(
            ((label, w / total) for label, w in weights.items()),
            key=lambda t: -t[1],
        )
        lines = [f"  {label}: vote share = {share:.3f}" for label, share in shares]
        return (
            f"Query's {self.k}-NN vote-share distribution among candidate "
            f"device types:\n" + "\n".join(lines) + "\n"
            f"How should this query be handled?"
        )

    def _call_ollama(self, prompt: str) -> dict:
        import json as _json
        import urllib.request

        payload = _json.dumps({
            "model": self.model,
            "system": self.SYSTEM_PROMPT,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0},
        }).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise LLMOnboardingPolicyError(
                f"Could not reach Ollama at {self.base_url} (is `ollama serve` "
                f"running, and has `ollama pull {self.model}` been run?): {e}"
            ) from e
        raw_text = body.get("response", "")
        try:
            return _json.loads(raw_text)
        except _json.JSONDecodeError as e:
            raise LLMOnboardingPolicyError(
                f"Model response was not valid JSON: {raw_text!r}"
            ) from e

    def decide(
        self, kb: DeviceFingerprintKB, query_embedding: np.ndarray
    ) -> Tuple[OnboardAction, str, float, str]:
        neighbors = kb.knn(query_embedding, self.k)
        if not neighbors:
            return "escalate", "no fingerprints in knowledge base yet", 0.0, ""

        weights = {}
        for entry, dist in neighbors:
            w = 1.0 / (dist + 1e-6)
            weights[entry.label] = weights.get(entry.label, 0.0) + w
        prompt = self._build_prompt(weights, neighbors)

        last_error: Exception = LLMOnboardingPolicyError("no attempts made")
        for _ in range(self.max_retries + 1):
            try:
                out = self._call_ollama(prompt)
                action = out["action"]
                label = str(out["predicted_label"])
                confidence = float(out["confidence"])
                rationale = str(out.get("rationale", ""))
                if action not in ("auto-admit", "request-more-data", "escalate"):
                    raise LLMOnboardingPolicyError(f"invalid action: {action!r}")
                if not (0.0 <= confidence <= 1.0):
                    raise LLMOnboardingPolicyError(f"confidence out of range: {confidence}")
                return action, rationale, confidence, label
            except (LLMOnboardingPolicyError, KeyError, ValueError, TypeError) as e:
                last_error = e
        raise LLMOnboardingPolicyError(
            f"Failed to get a valid decision after {self.max_retries + 1} attempts: {last_error}"
        )
