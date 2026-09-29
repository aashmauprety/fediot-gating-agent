"""Explainable intrusion triage (Section VI-B "Explainable Intrusion
Triage"): when the attack-type head flags traffic, retrieve reference
exemplars for the predicted class from an attack-signature knowledge base
and produce a rationale + a confirm/flag-for-review/likely-false-positive
triage action.

`RuleBasedTriagePolicy` is the documented fixed-margin stand-in.
`LLMTriagePolicy` (below) is a real implementation, backed by a local
Ollama server: it reasons over the SAME already-normalized ratio
(query's mean distance to same-class exemplars, relative to that
class's own typical intra-class spread) the rule-based policy computes,
plus calibration few-shot examples, to decide
confirm/flag-for-review/likely-false-positive with a real
natural-language rationale.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

import numpy as np

TriageAction = Literal["confirm", "flag-for-review", "likely-false-positive"]


@dataclass
class SignatureEntry:
    attack_class: str
    features: np.ndarray


class AttackSignatureKB:
    def __init__(self):
        self.entries: List[SignatureEntry] = []

    def build_from_calibration(self, X: np.ndarray, y: np.ndarray, class_names: List[str], max_per_class: int = 200):
        rng = np.random.RandomState(0)
        for c_idx, c_name in enumerate(class_names):
            idx = np.where(y == c_idx)[0]
            if len(idx) == 0:
                continue
            if len(idx) > max_per_class:
                idx = rng.choice(idx, size=max_per_class, replace=False)
            for i in idx:
                self.entries.append(SignatureEntry(c_name, X[i]))

    def knn(self, query: np.ndarray, attack_class: str, k: int = 5) -> List[Tuple[SignatureEntry, float]]:
        candidates = [e for e in self.entries if e.attack_class == attack_class]
        if not candidates:
            return []
        dists = [(e, float(np.linalg.norm(e.features - query))) for e in candidates]
        dists.sort(key=lambda t: t[1])
        return dists[:k]


class RuleBasedTriagePolicy:
    def __init__(self, k: int = 5, confirm_margin: float = 0.5, fp_margin: float = 2.0):
        self.k = k
        self.confirm_margin = confirm_margin
        self.fp_margin = fp_margin

    def decide(
        self, kb: AttackSignatureKB, query: np.ndarray, predicted_class: str
    ) -> Tuple[TriageAction, str]:
        neighbors = kb.knn(query, predicted_class, self.k)
        if not neighbors:
            return "flag-for-review", f"No reference exemplars for predicted class '{predicted_class}' in KB."
        mean_dist = float(np.mean([d for _, d in neighbors]))
        # compare to the KB's own internal spread for that class as a scale reference
        all_dists = [np.linalg.norm(a.features - b.features)
                     for a in kb.entries if a.attack_class == predicted_class
                     for b in kb.entries if b.attack_class == predicted_class][:200]
        typical = float(np.median(all_dists)) if all_dists else mean_dist
        ratio = mean_dist / (typical + 1e-6)

        rationale = (
            f"Query's mean distance to {self.k} nearest '{predicted_class}' exemplars = {mean_dist:.3f}, "
            f"vs. typical intra-class distance {typical:.3f} (ratio {ratio:.2f})."
        )
        if ratio <= self.confirm_margin + 1.0:
            return "confirm", rationale
        elif ratio <= self.fp_margin:
            return "flag-for-review", rationale
        else:
            return "likely-false-positive", rationale


class LLMTriagePolicyError(RuntimeError):
    """Raised on any failure to get a valid decision from the LLM
    backend. Never silently downgraded to RuleBasedTriagePolicy."""


class LLMTriagePolicy:
    """Real LLM-based triage agent, backed by a local Ollama server. Given
    the SAME distance-ratio signal RuleBasedTriagePolicy computes
    (already normalized -- a ratio near 1.0 means the flagged traffic
    looks like a typical example of the predicted class, not a raw,
    scale-dependent distance), it decides
    confirm/flag-for-review/likely-false-positive with a real
    natural-language rationale."""

    SYSTEM_PROMPT = (
        "You are an intrusion-triage agent for a federated IoT network. "
        "The attack-type classifier has flagged some traffic as a "
        "specific predicted attack class. You are given: the query's "
        "mean distance to its k nearest reference exemplars of that "
        "predicted class, and that class's own typical (median) "
        "intra-class distance among its reference exemplars -- expressed "
        "as a ratio (query distance / typical intra-class distance). A "
        "ratio near 1.0 means the flagged traffic looks like a typical "
        "example of the predicted class; a much higher ratio means it "
        "looks atypical for that class despite being labeled that way, "
        "suggesting the classifier may have made an error. Decide: "
        '"confirm" (the traffic closely resembles known exemplars of the '
        'predicted class -- the alert is credible), "flag-for-review" '
        "(a partial or borderline match -- worth a human look before "
        'acting), or "likely-false-positive" (does not resemble known '
        "exemplars of the predicted class at all -- probably a "
        "classifier error).\n\n"
        "Calibration reference:\n"
        "Example 1 (typical match): ratio = 1.1 -> confirm (close to the "
        "class's own typical spread, an ordinary example of this "
        "class).\n"
        "Example 2 (borderline): ratio = 1.8 -> flag-for-review (visibly "
        "farther than typical, but not wildly so -- ambiguous).\n"
        "Example 3 (clear mismatch): ratio = 3.5 -> likely-false-positive "
        "(much farther from this class's exemplars than that class's own "
        "members typically are from each other -- the label likely does "
        "not fit this traffic).\n\n"
        "Now judge the real case below the same way. Respond with ONLY "
        'a JSON object of the exact form {"action": "confirm" | '
        '"flag-for-review" | "likely-false-positive", "rationale": '
        '"<one sentence naming the specific ratio value that drove the '
        'decision>"}. No text before or after the JSON.'
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
            raise LLMTriagePolicyError(
                f"Could not reach Ollama at {self.base_url} (is `ollama serve` "
                f"running, and has `ollama pull {self.model}` been run?): {e}"
            ) from e
        raw_text = body.get("response", "")
        try:
            return _json.loads(raw_text)
        except _json.JSONDecodeError as e:
            raise LLMTriagePolicyError(
                f"Model response was not valid JSON: {raw_text!r}"
            ) from e

    def decide(
        self, kb: AttackSignatureKB, query: np.ndarray, predicted_class: str
    ) -> Tuple[TriageAction, str]:
        neighbors = kb.knn(query, predicted_class, self.k)
        if not neighbors:
            return "flag-for-review", f"No reference exemplars for predicted class '{predicted_class}' in KB."
        mean_dist = float(np.mean([d for _, d in neighbors]))
        all_dists = [np.linalg.norm(a.features - b.features)
                     for a in kb.entries if a.attack_class == predicted_class
                     for b in kb.entries if b.attack_class == predicted_class][:200]
        typical = float(np.median(all_dists)) if all_dists else mean_dist
        ratio = mean_dist / (typical + 1e-6)

        prompt = (
            f"Predicted class: {predicted_class}\n"
            f"Ratio (query's mean distance to {self.k} nearest exemplars "
            f"/ this class's own typical intra-class distance): {ratio:.3f}\n"
            f"How should this alert be triaged?"
        )

        last_error: Exception = LLMTriagePolicyError("no attempts made")
        for _ in range(self.max_retries + 1):
            try:
                out = self._call_ollama(prompt)
                action = out["action"]
                rationale = str(out.get("rationale", ""))
                if action not in ("confirm", "flag-for-review", "likely-false-positive"):
                    raise LLMTriagePolicyError(f"invalid action: {action!r}")
                return action, rationale
            except (LLMTriagePolicyError, KeyError, ValueError, TypeError) as e:
                last_error = e
        raise LLMTriagePolicyError(
            f"Failed to get a valid decision after {self.max_retries + 1} attempts: {last_error}"
        )
