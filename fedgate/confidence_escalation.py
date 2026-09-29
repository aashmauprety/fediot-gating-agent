"""Confidence-gated LLM escalation: the design the user actually asked
for (clarified mid-session, after this codebase had instead spent a long
stretch on the Gating Agent -- a federated-training-time defense
mechanism, a different thing). This is simpler and more directly suited
to what an LLM is good at: run the normal classifier as usual; when its
own top prediction is confident, just use it (no LLM call, no cost);
when it is NOT confident (probability mass spread across several
classes), hand the LLM the classifier's own probability distribution
over the top candidates and ask it to make the final call, with a
rationale -- a single-shot reasoning-over-evidence task, not the
multi-round statistical-trust-maintenance task the Gating Agent needed
and that six real experiments showed local LLMs are bad at.

Real, honest test protocol (see scripts/run_ciciot_diad_confidence_escalation.py):
split the test set into "confident" (classifier's top-1 probability >=
threshold) and "uncertain" (below it); on the uncertain subset
specifically, compare the classifier's own top-1 guess (baseline)
against the LLM's decision given the same probability distribution --
isolating whether the LLM adds real value exactly where it's supposed
to (uncertain cases), not conflating it with the confident majority
where any reasonable policy just matches the classifier.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


class ConfidenceEscalationError(RuntimeError):
    """Raised on any failure to get a valid decision from the LLM
    backend. Never silently falls back to the classifier's raw top-1 --
    a caller that wants a fallback must do so explicitly."""


@dataclass
class EscalationDecision:
    predicted_label: str
    action: str  # "confirm_top1" | "override" | "escalate_to_human"
    rationale: str


def rule_based_escalation(class_names: List[str], probs: "np.ndarray", escalate_below: float = 0.15) -> EscalationDecision:
    """Stand-in for comparison: just take the classifier's own top-1,
    or say "escalate_to_human" if even the top-1 is extremely low
    confidence. Does no real reasoning -- this is the naive baseline
    the LLM escalation is measured against, on the SAME uncertain
    subset."""
    import numpy as np
    top_idx = int(np.argmax(probs))
    top_p = float(probs[top_idx])
    if top_p < escalate_below:
        return EscalationDecision(class_names[top_idx], "escalate_to_human",
                                   f"top-1 probability {top_p:.2%} is below the escalation floor")
    return EscalationDecision(class_names[top_idx], "confirm_top1",
                               f"top-1 probability {top_p:.2%}, no reasoning applied")


# A hand-picked, human-interpretable subset of CIC IoT-DIAD 2024's 118
# numeric features to show the LLM verbatim (not all 118 -- too much to
# usefully include in one prompt, and many are opaque multi-window
# statistics rather than directly interpretable protocol/timing fields).
# Motivated by the real finding that probability-only escalation adds no
# value (Section on confidence-gated escalation): the classifier's
# output probabilities are a strict derivative of the features it was
# trained on, so an LLM given only those probabilities has no genuinely
# new evidence to reason from. These raw fields are things a person (or
# an LLM with real-world protocol knowledge) could reason about directly
# -- e.g. "port 1900 is SSDP, common on smart-hub discovery traffic" --
# in a way a re-derived probability number cannot convey.
DIAD_INTERPRETABLE_FEATURES = [
    "src_port", "dst_port", "port_class_dst", "l4_tcp", "l4_udp",
    "ttl", "eth_size", "tcp_window_size", "payload_entropy", "payload_length",
    "inter_arrival_time", "jitter", "http_response_code", "dns_query_type", "icmp_type",
]


class LLMFeatureAwareEscalationPolicy:
    """Extension of LLMConfidenceEscalationPolicy: gives the LLM real,
    named, raw (unscaled) feature values for the sample in question,
    alongside the classifier's probability distribution over top
    candidates -- genuinely new evidence beyond what the classifier's
    own output already encodes, unlike the probability-only version."""

    SYSTEM_PROMPT = (
        "You are helping decide a network traffic classification the "
        "primary classifier is NOT confident about. You will be shown "
        "(1) the classifier's own probability distribution over its top "
        "candidate classes, AND (2) real, named raw traffic feature "
        "values for this specific sample (ports, protocol flags, TTL, "
        "TCP window size, payload entropy/length, timing). Use your own "
        "knowledge of network protocols and typical IoT device behavior "
        "to reason about which candidate class these raw values are "
        "actually consistent with -- e.g. specific port numbers can "
        "indicate a known protocol or service (like SSDP, mDNS, a "
        "device's cloud API), unusual TTL or TCP window size can hint "
        "at device OS/hardware, and payload entropy can hint at "
        "encrypted vs. plaintext traffic. Do not simply repeat the "
        "highest-probability class by default -- use the raw features as "
        "real, independent evidence, not just a restatement of the "
        "probabilities you were already given. Respond with ONLY a JSON "
        "object of the exact form "
        '{"predicted_label": "<one of the exact candidate class names '
        'given, or "ESCALATE" if too ambiguous>", '
        '"rationale": "<one sentence, citing at least one specific raw '
        'feature value that drove the decision>"}. '
        "No text before or after the JSON."
    )

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        top_k: int = 5,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.top_k = top_k

    def _build_prompt(self, class_names, probs, feature_names, feature_values):
        import numpy as np
        order = np.argsort(-probs)[: self.top_k]
        candidates = [(class_names[i], float(probs[i])) for i in order]
        prob_lines = "\n".join(f"  {name}: {p:.1%}" for name, p in candidates)
        feat_lines = "\n".join(f"  {n} = {v:.4g}" for n, v in zip(feature_names, feature_values))
        prompt = (
            f"Classifier's top {len(candidates)} candidates, by probability:\n"
            f"{prob_lines}\n\n"
            f"Raw feature values for this sample:\n"
            f"{feat_lines}\n\n"
            f"Which class is correct, or should this be escalated?"
        )
        return prompt, [c[0] for c in candidates]

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
            raise ConfidenceEscalationError(
                f"Could not reach Ollama at {self.base_url}: {e}"
            ) from e
        raw_text = body.get("response", "")
        try:
            return _json.loads(raw_text)
        except _json.JSONDecodeError as e:
            raise ConfidenceEscalationError(f"Model response was not valid JSON: {raw_text!r}") from e

    def decide(self, class_names, probs, feature_names, feature_values) -> EscalationDecision:
        prompt, candidate_names = self._build_prompt(class_names, probs, feature_names, feature_values)
        last_error: Exception = ConfidenceEscalationError("no attempts made")
        for _ in range(self.max_retries + 1):
            try:
                out = self._call_ollama(prompt)
                label = str(out["predicted_label"])
                rationale = str(out.get("rationale", ""))
                if label == "ESCALATE":
                    return EscalationDecision(candidate_names[0], "escalate_to_human", rationale)
                if label not in candidate_names:
                    raise ConfidenceEscalationError(
                        f"model returned {label!r}, not one of the given candidates {candidate_names}"
                    )
                action = "confirm_top1" if label == candidate_names[0] else "override"
                return EscalationDecision(label, action, rationale)
            except (ConfidenceEscalationError, KeyError, ValueError, TypeError) as e:
                last_error = e
        raise ConfidenceEscalationError(f"Failed after {self.max_retries + 1} attempts: {last_error}")


class LLMConfidenceEscalationPolicy:
    """Real LLM-based escalation, backed by a local Ollama server (same
    infrastructure already validated for the Gating Agent experiments --
    no API key, no per-call billing). Given the classifier's own
    probability distribution over its top-k candidate classes for one
    uncertain sample, asks the model to pick the final label (confirming
    the classifier's top-1, or overriding it in favor of a different
    candidate) or explicitly escalate to a human if genuinely too
    ambiguous to call."""

    SYSTEM_PROMPT = (
        "You are helping decide a network traffic classification the "
        "primary classifier is NOT confident about. You will be shown "
        "the classifier's own probability distribution over its top "
        "candidate classes for one sample (not the raw traffic itself, "
        "just the classifier's confidence in each candidate). Your job: "
        "decide which class is most likely correct, or say the case "
        "should be escalated to a human if the evidence is genuinely too "
        "ambiguous to call (e.g. the top candidates are very close in "
        "probability with no clear front-runner). Do not simply repeat "
        "the highest-probability class by default -- consider whether "
        "the specific class names and their relative probabilities "
        "suggest a more likely answer (e.g. two visually/behaviorally "
        "similar device or attack types being confused for each other is "
        "common and worth reasoning about explicitly). Respond with ONLY "
        "a JSON object of the exact form "
        '{"predicted_label": "<one of the exact candidate class names '
        'given, or "ESCALATE" if too ambiguous>", '
        '"rationale": "<one sentence explaining the decision>"}. '
        "No text before or after the JSON."
    )

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        top_k: int = 5,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.top_k = top_k

    def _build_prompt(self, class_names: List[str], probs: "np.ndarray") -> Tuple[str, List[str]]:
        import numpy as np
        order = np.argsort(-probs)[: self.top_k]
        candidates = [(class_names[i], float(probs[i])) for i in order]
        lines = "\n".join(f"  {name}: {p:.1%}" for name, p in candidates)
        prompt = (
            f"Classifier's top {len(candidates)} candidates for this sample, by probability:\n"
            f"{lines}\n"
            f"Which class is correct, or should this be escalated?"
        )
        return prompt, [c[0] for c in candidates]

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
            raise ConfidenceEscalationError(
                f"Could not reach Ollama at {self.base_url} (is `ollama serve` "
                f"running, and has `ollama pull {self.model}` been run?): {e}"
            ) from e
        raw_text = body.get("response", "")
        try:
            return _json.loads(raw_text)
        except _json.JSONDecodeError as e:
            raise ConfidenceEscalationError(f"Model response was not valid JSON: {raw_text!r}") from e

    def decide(self, class_names: List[str], probs: "np.ndarray") -> EscalationDecision:
        prompt, candidate_names = self._build_prompt(class_names, probs)
        last_error: Exception = ConfidenceEscalationError("no attempts made")
        for _ in range(self.max_retries + 1):
            try:
                out = self._call_ollama(prompt)
                label = str(out["predicted_label"])
                rationale = str(out.get("rationale", ""))
                if label == "ESCALATE":
                    return EscalationDecision(candidate_names[0], "escalate_to_human", rationale)
                if label not in candidate_names:
                    raise ConfidenceEscalationError(
                        f"model returned {label!r}, not one of the given candidates {candidate_names}"
                    )
                action = "confirm_top1" if label == candidate_names[0] else "override"
                return EscalationDecision(label, action, rationale)
            except (ConfidenceEscalationError, KeyError, ValueError, TypeError) as e:
                last_error = e
        raise ConfidenceEscalationError(f"Failed after {self.max_retries + 1} attempts: {last_error}")
