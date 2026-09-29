"""The Gating Agent (Section VI-B): digest computation, the Bayesian
long-term trust prior (Section VI-B1), and a gating policy.

HONESTY NOTE: `RuleBasedGatingPolicy` is an explicit, documented stand-in
for the paper's LLM-based reasoning step, used for every result reported
in Section VII of the paper so far. It makes the same three-way decision
(accept/downweight/exclude) from the same digest + Bayesian-posterior
inputs the paper describes the LLM agent consuming, using fixed
thresholds instead of free-form reasoning, and it does NOT call
FetchHistory/FetchSecondaryCalibration as separate tool invocations --
it just always looks at the posterior (there is no ambiguity-triggered
multi-step behavior here, since there is no LLM to decide "am I
ambiguous"). Results produced with this policy support an ablation
("digest+Bayesian gating, rule-based" vs "plain FedAvg" vs "static
robust-aggregation baselines") but must NOT be reported as validating the
LLM-agent design itself.

`LLMGatingPolicy` below is now a real implementation, backed by a local
Ollama server (no API key or per-call billing) -- see
code/scripts/smoke_test_llm_gating.py for a minimal usage check and
code/README.md for setup instructions. It has NOT yet been used to
produce any result in the paper; doing so, and comparing against
RuleBasedGatingPolicy, is the clearest remaining open experiment.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

Action = Literal["accept", "downweight", "exclude"]


def describe_calib_delta(delta: float) -> str:
    """Worded direction instead of a signed number -- a model that
    sometimes reads a negative delta as suspicious (Section VII-E
    "tenth real result") cannot get the direction backwards if the
    direction is stated in words rather than inferred from a sign.
    Shared by every LLM-based policy's prompt construction."""
    if delta > 0.1:
        return f"got much worse (held-out loss increased by {delta:.4f})"
    if delta > 0.01:
        return f"got slightly worse (held-out loss increased by {delta:.4f})"
    if delta > -0.01:
        return "essentially unchanged"
    if delta > -0.1:
        return f"got slightly better (held-out loss decreased by {-delta:.4f})"
    return f"got much better (held-out loss decreased by {-delta:.4f})"


def cosine_zscore(cosine: float, round_cosines: Optional[List[float]]) -> Optional[float]:
    """Same-round cross-client z-score for cosine similarity (Section
    VII-E "tenth/eleventh real result") -- cancels out a round-wide
    shift (e.g. cosine trending lower for everyone as training
    converges) the same way update_norm's z-score already cancels out
    round-to-round scale changes. Returns None (caller falls back to
    the raw value) if fewer than 2 other gateways' cosines are given."""
    if not round_cosines or len(round_cosines) < 2:
        return None
    arr = np.array(round_cosines)
    mu, sigma = arr.mean(), arr.std() + 1e-8
    return (cosine - mu) / sigma


def early_round_leniency_scale(
    round_idx: int, warmup_rounds: int = 5, max_scale: float = 1.5
) -> float:
    """Section VII-E fix #2: early-training digests are genuinely more
    volatile (the global model's update direction hasn't stabilized yet),
    so judging round 1 against the same fixed threshold as round 20
    mistakes ordinary early volatility for an attack. Returns a
    multiplier >= 1.0 that widens exclude/downweight thresholds, linearly
    interpolating from `max_scale` at round 0 down to `1.0` at
    `warmup_rounds` and beyond (round 0 itself is normally handled
    separately -- see fix #1 in experiment.py -- but this still applies
    if a caller invokes a policy directly on round 0)."""
    if warmup_rounds <= 0 or round_idx >= warmup_rounds:
        return 1.0
    frac = round_idx / warmup_rounds
    return max_scale - frac * (max_scale - 1.0)


@dataclass
class Digest:
    client_id: str
    component: str
    update_norm: float
    cosine_sim_to_prev_update: float
    calib_loss_delta: float  # positive = loss got worse vs. previous global model


@dataclass
class BayesianTrust:
    """Per (client, component) Beta-Bernoulli posterior, Section VI-B1.

    `decay` (Section VII-E fix #3): with the default `decay=1.0`, this is
    an unbroken all-time average, which has no recovery path -- one
    wrongly-excluded early round permanently caps how high the posterior
    mean can climb, since every later good round only dilutes a fixed
    amount of bad evidence rather than displacing it. `decay < 1.0`
    shrinks the existing (alpha, beta) mass toward the Beta(1,1) prior by
    that factor before adding each new observation, so old evidence
    fades geometrically and recent rounds dominate -- a client that was
    wrongly penalized early can recover to a high posterior mean given
    enough subsequent good rounds, instead of being permanently capped."""

    alpha: Dict[Tuple[str, str], float] = field(default_factory=dict)
    beta: Dict[Tuple[str, str], float] = field(default_factory=dict)
    decay: float = 1.0

    def _key(self, client_id: str, component: str) -> Tuple[str, str]:
        return (client_id, component)

    def mean(self, client_id: str, component: str) -> float:
        k = self._key(client_id, component)
        a, b = self.alpha.get(k, 1.0), self.beta.get(k, 1.0)
        return a / (a + b)

    def credible_interval(self, client_id: str, component: str, z: float = 1.0):
        """A cheap normal-approximation interval around the Beta mean,
        not a true Beta quantile -- sufficient for reporting/rationale
        purposes, not for anything requiring exact coverage guarantees."""
        k = self._key(client_id, component)
        a, b = self.alpha.get(k, 1.0), self.beta.get(k, 1.0)
        n = a + b
        mean = a / n
        var = (a * b) / (n * n * (n + 1))
        sd = var ** 0.5
        return max(0.0, mean - z * sd), min(1.0, mean + z * sd)

    def update(self, client_id: str, component: str, gating_score: float) -> None:
        """Eq. in Section VI-B1: tau treated as a soft observation, with
        the existing mass decayed toward the prior first (no-op when
        `decay == 1.0`, i.e. the original unbroken-average behavior)."""
        k = self._key(client_id, component)
        a = self.alpha.get(k, 1.0)
        b = self.beta.get(k, 1.0)
        if self.decay != 1.0:
            a = 1.0 + (a - 1.0) * self.decay
            b = 1.0 + (b - 1.0) * self.decay
        self.alpha[k] = a + gating_score
        self.beta[k] = b + (1.0 - gating_score)

    def n_observations(self, client_id: str, component: str) -> float:
        k = self._key(client_id, component)
        return self.alpha.get(k, 1.0) + self.beta.get(k, 1.0) - 2.0  # minus the Beta(1,1) prior mass


@dataclass
class GatingDecision:
    client_id: str
    component: str
    gating_score: float  # tau in [0,1]
    action: Action
    rationale: str


class RuleBasedGatingPolicy:
    """Deterministic stand-in for the LLM Gating Agent's per-round
    decision. See module docstring for the honesty caveat.

    Decision rule (documented, not learned): combine this round's digest
    anomaly score with the long-term Bayesian posterior mean. A client
    with a strong long-term posterior is given the benefit of the doubt on
    a borderline round (downweight rather than exclude); a client with a
    weak/short posterior is judged mostly on the current round.
    """

    def __init__(
        self,
        norm_exclude_z: float = 3.0,
        norm_downweight_z: float = 1.5,
        cosine_exclude: float = -0.2,
        calib_exclude_delta: float = 0.5,
        use_bayesian_prior: bool = True,
        warmup_rounds: int = 0,
        warmup_scale: float = 1.5,
    ):
        self.norm_exclude_z = norm_exclude_z
        self.norm_downweight_z = norm_downweight_z
        self.cosine_exclude = cosine_exclude
        self.calib_exclude_delta = calib_exclude_delta
        # Ablation switch (Section VII-E "round-only" variant): when False,
        # the policy still *updates* the Bayesian posterior every round
        # (so it can be inspected/reported), but never *consults* it when
        # deciding -- isolating what the long-term prior itself
        # contributes versus the round-level digest alone.
        self.use_bayesian_prior = use_bayesian_prior
        # Section VII-E fix #2: widen the norm thresholds for the first
        # `warmup_rounds` rounds (0 = disabled, the original behavior).
        self.warmup_rounds = warmup_rounds
        self.warmup_scale = warmup_scale

    def decide(
        self,
        digest: Digest,
        round_update_norms: List[float],
        trust: BayesianTrust,
        round_idx: int = -1,
        round_cosines: Optional[List[float]] = None,  # unused here; accepted for a uniform policy interface
        round_calib_deltas: Optional[List[float]] = None,  # unused here; accepted for a uniform policy interface
    ) -> GatingDecision:
        norms = np.array(round_update_norms)
        mu, sigma = norms.mean(), norms.std() + 1e-8
        z = (digest.update_norm - mu) / sigma

        scale = early_round_leniency_scale(round_idx, self.warmup_rounds, self.warmup_scale) \
            if round_idx >= 0 else 1.0
        norm_exclude_z = self.norm_exclude_z * scale
        norm_downweight_z = self.norm_downweight_z * scale

        posterior_mean = trust.mean(digest.client_id, digest.component)
        n_obs = trust.n_observations(digest.client_id, digest.component)
        long_term_confident = self.use_bayesian_prior and n_obs >= 5 and posterior_mean >= 0.7

        reasons = [f"update-norm z-score={z:.2f}"]
        if scale != 1.0:
            reasons.append(f"early-round warmup active (scale={scale:.2f})")

        # hard exclude conditions: obviously poisoned this round
        if digest.cosine_sim_to_prev_update < self.cosine_exclude or \
           digest.calib_loss_delta > self.calib_exclude_delta:
            reasons.append(
                f"cosine={digest.cosine_sim_to_prev_update:.2f}, "
                f"calib_loss_delta={digest.calib_loss_delta:.3f} both exceed exclude thresholds"
            )
            score = max(0.0, 0.1 * posterior_mean if long_term_confident else 0.0)
            action: Action = "exclude" if not long_term_confident else "downweight"
            if action == "downweight":
                reasons.append(
                    f"long-term posterior mean={posterior_mean:.2f} over {n_obs:.0f} rounds "
                    "downgrades exclude to downweight"
                )
            trust.update(digest.client_id, digest.component, score)
            return GatingDecision(digest.client_id, digest.component, score, action, "; ".join(reasons))

        if z > norm_exclude_z:
            if long_term_confident:
                score = 0.5
                action = "downweight"
                reasons.append(
                    f"norm z={z:.2f} > exclude threshold ({norm_exclude_z:.2f}), but long-term posterior "
                    f"mean={posterior_mean:.2f} over {n_obs:.0f} rounds -> downweight not exclude"
                )
            else:
                score = 0.0
                action = "exclude"
                reasons.append(f"norm z={z:.2f} > exclude threshold ({norm_exclude_z:.2f}), no strong history to offset it")
        elif z > norm_downweight_z:
            score = 0.5
            action = "downweight"
            reasons.append(f"norm z={z:.2f} > downweight threshold ({norm_downweight_z:.2f})")
        else:
            score = 1.0
            action = "accept"
            reasons.append("digest within normal range")

        trust.update(digest.client_id, digest.component, score)
        return GatingDecision(digest.client_id, digest.component, score, action, "; ".join(reasons))


def robust_zscore(value: float, population: List[float]) -> float:
    """Median/MAD z-score (robust to a minority of outliers skewing a
    plain mean/std), scaled so a normal distribution's MAD matches its
    std (the standard 1.4826 constant)."""
    arr = np.array(population)
    median = np.median(arr)
    mad = np.median(np.abs(arr - median)) * 1.4826
    return (value - median) / (mad + 1e-8)


class AdaptiveRuleBasedGatingPolicy(RuleBasedGatingPolicy):
    """Identical to `RuleBasedGatingPolicy` except the calibration-loss
    hard-exclude condition is judged as a SAME-ROUND, cross-client robust
    z-score (median/MAD across this round's `round_calib_deltas`) instead
    of a fixed absolute magnitude -- exactly the same trick the digest's
    update-norm and (in the LLM policies) cosine-similarity signals
    already use to cancel a round-wide shift structurally (Section III-A).

    An EWMA-history-based adaptive baseline was tried first and rejected:
    it has a genuine, real failure mode under a SUSTAINED linear drift --
    once a round is flagged anomalous, a naive "only update on accept"
    rule freezes the baseline, the gap to the still-drifting true value
    only grows, and the estimator can never recover (verified with
    `scripts/run_concept_drift_test.py`'s first version: 100% false-positive
    rate in the last 5 rounds, WORSE than the fixed threshold it was meant
    to fix). A same-round cross-client comparison has no such history
    dependency: genuine concept drift moves EVERY client's calib_loss_delta
    together, so the median moves with it and an honest client's z-score
    stays near 0 regardless of the drift's absolute magnitude, while a real
    (still-minority) attacker's update still stands out from its peers.
    """

    def __init__(self, *args, calib_exclude_z: float = 3.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.calib_exclude_z = calib_exclude_z

    def decide(
        self,
        digest: Digest,
        round_update_norms: List[float],
        trust: BayesianTrust,
        round_idx: int = -1,
        round_cosines: Optional[List[float]] = None,
        round_calib_deltas: Optional[List[float]] = None,
    ) -> GatingDecision:
        norms = np.array(round_update_norms)
        mu, sigma = norms.mean(), norms.std() + 1e-8
        z = (digest.update_norm - mu) / sigma

        if round_calib_deltas and len(round_calib_deltas) > 1:
            calib_z = robust_zscore(digest.calib_loss_delta, round_calib_deltas)
        else:
            calib_z = 0.0  # can't compute a cross-client z-score alone; fail open, not closed

        scale = early_round_leniency_scale(round_idx, self.warmup_rounds, self.warmup_scale) \
            if round_idx >= 0 else 1.0
        norm_exclude_z = self.norm_exclude_z * scale
        norm_downweight_z = self.norm_downweight_z * scale

        posterior_mean = trust.mean(digest.client_id, digest.component)
        n_obs = trust.n_observations(digest.client_id, digest.component)
        long_term_confident = self.use_bayesian_prior and n_obs >= 5 and posterior_mean >= 0.7

        reasons = [f"update-norm z-score={z:.2f}", f"calib-loss same-round z-score={calib_z:.2f}"]
        if scale != 1.0:
            reasons.append(f"early-round warmup active (scale={scale:.2f})")

        if digest.cosine_sim_to_prev_update < self.cosine_exclude or calib_z > self.calib_exclude_z:
            reasons.append(
                f"cosine={digest.cosine_sim_to_prev_update:.2f}, calib-loss same-round z={calib_z:.2f} "
                "exceed exclude thresholds"
            )
            score = max(0.0, 0.1 * posterior_mean if long_term_confident else 0.0)
            action: Action = "exclude" if not long_term_confident else "downweight"
            if action == "downweight":
                reasons.append(
                    f"long-term posterior mean={posterior_mean:.2f} over {n_obs:.0f} rounds "
                    "downgrades exclude to downweight"
                )
            trust.update(digest.client_id, digest.component, score)
            return GatingDecision(digest.client_id, digest.component, score, action, "; ".join(reasons))

        if z > norm_exclude_z:
            if long_term_confident:
                score, action = 0.5, "downweight"
                reasons.append(f"norm z={z:.2f} > exclude threshold, long-term posterior downgrades to downweight")
            else:
                score, action = 0.0, "exclude"
                reasons.append(f"norm z={z:.2f} > exclude threshold, no strong history to offset it")
        elif z > norm_downweight_z:
            score, action = 0.5, "downweight"
            reasons.append(f"norm z={z:.2f} > downweight threshold")
        else:
            score, action = 1.0, "accept"
            reasons.append("digest within same-round normal range")

        trust.update(digest.client_id, digest.component, score)
        return GatingDecision(digest.client_id, digest.component, score, action, "; ".join(reasons))


class LLMGatingPolicyError(RuntimeError):
    """Raised on any failure to get a valid decision from the LLM backend
    (server unreachable, malformed/unparseable response, out-of-range
    values, etc). Deliberately not caught and silently downgraded to
    RuleBasedGatingPolicy anywhere -- a caller that wants a fallback must
    do so explicitly, so results are never mislabeled as LLM-gated when
    they were not."""


class LLMGatingPolicy:
    """Real LLM-based Gating Agent, backed by a local Ollama server
    (https://ollama.com) -- no external API key or per-call billing
    required, consistent with this paper's eventual "constrained edge
    gateway" deployment story (Section IX). Any Ollama-compatible model
    works; this was written against `llama3.1:8b`.

    Given the exact same inputs as RuleBasedGatingPolicy (a Digest, this
    round's update norms across clients, and the BayesianTrust posterior),
    it prompts the model to make the same three-way accept/downweight/
    exclude decision, but by actual reasoning over the evidence rather
    than fixed thresholds, and returns a short natural-language rationale
    generated by the model itself (not a templated string).

    Two modes:

    - `use_history_tool=False` (default, matches the first version of this
      class): single-turn, no tool use. Format the evidence, ask for a
      decision. This is what produced the Section VII-E result showing a
      D=10 (trust-building, 10-round delay) collapse -- the model appeared
      to react to each round's surface anomaly rather than the long-term
      posterior.
    - `use_history_tool=True`: implements the paper's FetchHistory step
      (Section VI-B). The model's first turn may respond with
      `{"action": "fetch_history"}` instead of a decision; if it does, we
      supply the gateway's actual last-K decision history (not just the
      single scalar posterior mean) and require a final decision on a
      second turn. This targets the diagnosed D=10 failure mode directly:
      a scalar posterior mean is easy to under-weight against one
      alarming-looking round, but an explicit trajectory ("accepted the
      last 9 rounds in a row, first anomaly this round") is harder to
      ignore.
    """

    # Section VII-E fix (same one applied to LLMExternalBayesianPolicy,
    # "tenth/eleventh/twelfth real result"): cosine is described as a
    # same-round cross-client z-score (cancels a round-wide shift the
    # same way update-norm's z-score already does) and calibration loss
    # as a worded direction (so the sign cannot be misread), instead of
    # a raw cosine value and a signed calib_delta number.
    SYSTEM_PROMPT_NO_TOOL = (
        "You are the Gating Agent in a federated learning system for IoT "
        "device identification and intrusion detection. Each round, "
        "participating gateways submit model updates that get averaged "
        "together. A fraction of gateways may be compromised and submit "
        "poisoned updates -- either random sabotage (untargeted attacks), "
        "updates crafted to mislabel one specific class (targeted "
        "attacks), or updates from a gateway that behaved honestly for "
        "many rounds before switching to an attack (trust-building "
        "attacks). Your job is to decide, for one gateway's update this "
        "round, whether to accept it fully, downweight it (use it but "
        "count it less in the average), or exclude it entirely. You are "
        "given: (1) this round's digest for the gateway -- its update "
        "norm's z-score vs. the other gateways submitting this round, "
        "its cosine similarity's z-score vs. those SAME gateways this "
        "SAME round (both z-scores are already relative to this round's "
        "own group, so a shift affecting every gateway equally this "
        "round is already cancelled out -- judge deviation from this "
        "round's group, not from some fixed absolute number or from a "
        "different round), and a worded description of how its "
        "held-out calibration loss changed; and (2) the gateway's "
        "long-term trust posterior -- a probability, built from many "
        "past rounds, that this gateway behaves normally. Weigh "
        "long-term consistency, not just this single round, since an "
        "attacker can make one round look normal. Respond with ONLY a "
        'JSON object of the exact form {"action": "accept" | '
        '"downweight" | "exclude", "gating_score": <number between 0.0 '
        "and 1.0, 1.0 = fully trust this round's update, 0.0 = fully "
        'discard it>, "rationale": "<one sentence, naming the specific '
        'evidence that drove the decision>"}. No text before or after '
        "the JSON."
    )

    SYSTEM_PROMPT_WITH_TOOL = (
        "You are the Gating Agent in a federated learning system for IoT "
        "device identification and intrusion detection. Each round, "
        "participating gateways submit model updates that get averaged "
        "together. A fraction of gateways may be compromised and submit "
        "poisoned updates -- either random sabotage (untargeted attacks), "
        "updates crafted to mislabel one specific class (targeted "
        "attacks), or updates from a gateway that behaved honestly for "
        "many rounds before switching to an attack (trust-building "
        "attacks, the hardest case: one anomalous round after a long "
        "honest history should usually NOT be excluded outright, since "
        "sustained good behavior is real evidence, but a pattern of "
        "worsening rounds should be). Your job is to decide, for one "
        "gateway's update this round, whether to accept it fully, "
        "downweight it, or exclude it entirely. You are given this "
        "round's digest -- update norm's z-score vs. other gateways this "
        "round, cosine similarity's z-score vs. those SAME gateways this "
        "SAME round (both already relative to this round's own group, so "
        "a shift affecting every gateway equally this round, e.g. cosine "
        "trending lower for everyone as training converges, is already "
        "cancelled out), and a worded description of the calibration-loss "
        "change -- and a long-term trust posterior (mean and number of "
        "observed rounds). If the posterior alone is not enough to judge "
        "whether this round's anomaly is a one-off against a long "
        "history of good behavior or part of a real pattern, you may "
        'respond with EXACTLY {"action": "fetch_history"} to request '
        "that gateway's actual recent per-round decision record before "
        "deciding -- do this when the gateway has a high posterior mean "
        "but this round looks anomalous, since that is exactly the "
        "ambiguous case history resolves. Otherwise, or after you "
        "receive the history, respond with ONLY a JSON object of the "
        'exact form {"action": "accept" | "downweight" | "exclude", '
        '"gating_score": <number between 0.0 and 1.0, 1.0 = fully trust '
        "this round's update, 0.0 = fully discard it>, "
        '"rationale": "<one sentence, naming the specific evidence that '
        'drove the decision>"}. No text before or after the JSON.'
    )

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        use_history_tool: bool = False,
        history_window: int = 10,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.use_history_tool = use_history_tool
        self.history_window = history_window
        # (client_id, component) -> list of past round records, most
        # recent last. Populated by this policy itself as it decides, so
        # FetchHistory reflects exactly what this policy has observed.
        self._history: Dict[Tuple[str, str], List[dict]] = {}

    def _build_prompt(
        self,
        digest: Digest,
        round_update_norms: List[float],
        posterior_mean: float,
        n_obs: float,
        round_cosines: Optional[List[float]] = None,
        round_idx: int = -1,
    ) -> str:
        norms = np.array(round_update_norms)
        mu, sigma = norms.mean(), norms.std() + 1e-8
        z = (digest.update_norm - mu) / sigma

        cos_z = cosine_zscore(digest.cosine_sim_to_prev_update, round_cosines)
        if cos_z is not None:
            cos_line = f"  cosine similarity z-score vs. the {len(round_cosines)} gateways submitting this round: {cos_z:+.2f}\n"
        else:
            cos_line = f"  cosine similarity to previous global update = {digest.cosine_sim_to_prev_update:.3f}\n"

        round_line = f"Training round: {round_idx}\n" if round_idx >= 0 else ""
        return (
            f"{round_line}"
            f"Gateway: {digest.client_id}, component: {digest.component}\n"
            f"This round's digest:\n"
            f"  update norm z-score vs. the {len(round_update_norms)} gateways submitting this round: {z:+.2f}\n"
            f"{cos_line}"
            f"  calibration loss: {describe_calib_delta(digest.calib_loss_delta)}\n"
            f"Long-term trust posterior for this gateway: mean="
            f"{posterior_mean:.3f} over {n_obs:.0f} observed rounds "
            f"(0 rounds = no history yet, treat with caution either way).\n"
            f"Decide: accept, downweight, or exclude this update."
        )

    def _format_history(self, records: List[dict]) -> str:
        if not records:
            return "(no prior rounds recorded for this gateway yet)"
        lines = [
            f"  round {r['round_idx']}: action={r['action']}, gating_score={r['gating_score']:.2f}, "
            f"update_norm_z={r['z']:+.2f}, cosine_z={r['cosine_z']}, "
            f"calib_loss: {describe_calib_delta(r['calib_delta'])}"
            for r in records
        ]
        return "\n".join(lines)

    def _call_ollama(self, prompt: str, system_prompt: str) -> dict:
        import json as _json
        import urllib.request

        payload = _json.dumps({
            "model": self.model,
            "system": system_prompt,
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
            raise LLMGatingPolicyError(
                f"Could not reach Ollama at {self.base_url} (is `ollama serve` "
                f"running, and has `ollama pull {self.model}` been run?): {e}"
            ) from e

        raw_text = body.get("response", "")
        try:
            decision = _json.loads(raw_text)
        except _json.JSONDecodeError as e:
            raise LLMGatingPolicyError(
                f"Model response was not valid JSON: {raw_text!r}"
            ) from e
        return decision

    def decide(
        self,
        digest: Digest,
        round_update_norms: List[float],
        trust: "BayesianTrust",
        round_idx: int = -1,
        round_cosines: Optional[List[float]] = None,
        round_calib_deltas: Optional[List[float]] = None,  # unused here; accepted for a uniform policy interface
    ) -> GatingDecision:
        posterior_mean = trust.mean(digest.client_id, digest.component)
        n_obs = trust.n_observations(digest.client_id, digest.component)
        norms = np.array(round_update_norms)
        mu, sigma = norms.mean(), norms.std() + 1e-8
        z = (digest.update_norm - mu) / sigma
        cos_z = cosine_zscore(digest.cosine_sim_to_prev_update, round_cosines)
        prompt = self._build_prompt(
            digest, round_update_norms, posterior_mean, n_obs,
            round_cosines=round_cosines, round_idx=round_idx,
        )
        system_prompt = self.SYSTEM_PROMPT_WITH_TOOL if self.use_history_tool else self.SYSTEM_PROMPT_NO_TOOL
        hist_key = (digest.client_id, digest.component)

        last_error: Exception = LLMGatingPolicyError("no attempts made")
        for attempt in range(self.max_retries + 1):
            try:
                decision = self._call_ollama(prompt, system_prompt)

                if self.use_history_tool and decision.get("action") == "fetch_history":
                    history = self._history.get(hist_key, [])[-self.history_window:]
                    followup = (
                        prompt + "\n\nYou requested this gateway's recent history. "
                        f"Last {len(history)} round(s) (oldest first):\n"
                        f"{self._format_history(history)}\n"
                        "Now give your final decision as the required JSON object "
                        '(you may not request history again this round).'
                    )
                    decision = self._call_ollama(followup, system_prompt)

                action = decision["action"]
                score = float(decision["gating_score"])
                rationale = str(decision.get("rationale", ""))
                if action not in ("accept", "downweight", "exclude"):
                    raise LLMGatingPolicyError(f"invalid action: {action!r}")
                if not (0.0 <= score <= 1.0):
                    raise LLMGatingPolicyError(f"gating_score out of range: {score}")

                trust.update(digest.client_id, digest.component, score)
                self._history.setdefault(hist_key, []).append({
                    "round_idx": round_idx, "action": action, "gating_score": score,
                    "z": float(z),
                    "cosine_z": f"{cos_z:+.2f}" if cos_z is not None else "n/a (no round context)",
                    "calib_delta": digest.calib_loss_delta,
                })
                return GatingDecision(digest.client_id, digest.component, score, action, rationale)
            except (LLMGatingPolicyError, KeyError, ValueError, TypeError) as e:
                last_error = e
        raise LLMGatingPolicyError(
            f"Failed to get a valid decision after {self.max_retries + 1} attempts: {last_error}"
        )


class LLMExternalBayesianPolicy:
    """Hybrid architecture, motivated by a documented real failure of the
    pure single-call `LLMGatingPolicy` (Section VII-E of the paper):
    three real configurations (single-turn Llama 3.1 8B, + FetchHistory
    tool, Qwen2.5 14B single-turn) all failed to reliably weigh long-term
    trust against a single anomalous round -- most sharply, all three
    under-protected clients with a long clean history once a
    trust-building attacker triggered at $D=10$.

    Design, grounded in two literature threads that independently
    converged on the same diagnosis (frontier LLM-agent work arguing
    chain-of-thought "accumulates text but not probability mass," and FL
    trust literature proposing belief-state maintenance stay external to
    the LLM): the LLM here is given ONLY this round's digest -- no trust
    posterior, no history, no memory of any kind -- and asked for exactly
    one narrow judgment: how anomalous does this single round's digest
    look, as a score in [0,1]. It cannot "forget" long-term history
    because it is never given any; that state lives entirely outside the
    LLM, in the same deterministic `BayesianTrust` object and the same
    long-term-override decision structure `RuleBasedGatingPolicy` already
    uses. The LLM's anomaly judgment simply substitutes for the
    z-score-based per-round anomaly signal the rule-based policy computes
    from fixed thresholds; everything downstream -- the long-term-trust
    override, the posterior update, the final accept/downweight/exclude
    decision -- is identical in structure, deterministic, and entirely
    outside the LLM's control. Only the natural-language rationale
    attached to the decision comes from the model.
    """

    SYSTEM_PROMPT = (
        "You are a network anomaly judge in a federated learning system. "
        "You will be shown ONE round's statistical digest for one "
        "gateway's model update: its update norm and z-score vs. other "
        "gateways this round, its cosine similarity to the previous "
        "global update and ITS z-score vs. other gateways this same "
        "round, and a worded description of its calibration-loss change. "
        "Both z-scores are already computed relative to this round's "
        "other gateways, so a round-wide shift (e.g. cosine drifting "
        "lower for everyone as training converges) is already factored "
        "out -- judge deviation from THIS round's group, not from some "
        "fixed absolute number. Judge ONLY how anomalous this single "
        "round's digest looks, using no other context -- you are not "
        "told anything about this gateway's past behavior, and should "
        "not assume any. Respond with ONLY a JSON object of the exact "
        'form {"anomaly_score": <number between 0.0 (looks completely '
        'normal) and 1.0 (looks highly anomalous)>, "rationale": "<one '
        'sentence naming the specific digest value(s) that drove the '
        'score>"}. No text before or after the JSON.'
    )

    # Calibration fix (Section VII-E TODO item 8): the uncalibrated prompt
    # above gave the model no reference for what a "normal" z-score,
    # cosine similarity, or calibration delta actually looks like in this
    # system, and it defaulted to over-reporting anomaly on essentially
    # every round -- driving every gating weight to zero and silently
    # degrading the whole run into a safety-fallback plain FedAvg. This
    # variant adds three worked examples spanning the scale (clearly
    # normal, borderline, clearly anomalous) so the model has a concrete
    # anchor instead of self-calibrating from nothing.
    #
    # Revision (Section VII-E "tenth real result"): the first version of
    # this prompt gave cosine as a raw value (anchored at ~0.94 = normal)
    # plus a *textual* caveat that it drifts lower in later rounds -- the
    # caveat had no effect; the model still called any low raw cosine
    # anomalous regardless of round, and separately sometimes read
    # calib_loss_delta's sign backwards, treating a negative (improving)
    # delta as suspicious. Both digest fields are now expressed the same
    # way update_norm already was: as a same-round cross-client z-score
    # (which auto-cancels a round-wide shift, unlike a fixed threshold or
    # a prose caveat) for cosine, and as a worded direction ("got worse
    # by X" / "got better by X") for calibration loss instead of a signed
    # number a model can misread.
    SYSTEM_PROMPT_CALIBRATED = (
        "You are a network anomaly judge in a federated learning system. "
        "You will be shown ONE round's statistical digest for one "
        "gateway's model update: its update norm's z-score vs. other "
        "gateways this round, its cosine similarity's z-score vs. other "
        "gateways this SAME round, and a worded description of how its "
        "held-out calibration loss changed. Both z-scores already "
        "measure deviation from this round's own group, so a shift that "
        "affects every gateway equally this round (e.g. cosine "
        "similarity trending lower for everyone as training converges) "
        "is already cancelled out -- a z-score near 0 means normal for "
        "THIS round, regardless of what round it is or what the raw "
        "value would have been. Judge ONLY how anomalous this single "
        "round's digest looks, using no other context -- you are not "
        "told anything about this gateway's past behavior, and should "
        "not assume any.\n\n"
        "Calibration reference -- most rounds from honest gateways look "
        "like Example 1. Only score above 0.5 when the digest clearly "
        "resembles Example 3, not merely because a value is nonzero:\n"
        "Example 1 (typical honest round): update_norm_z=+0.15, "
        "cosine_z=+0.10, calibration loss: essentially unchanged -> "
        "anomaly_score=0.05 (both z-scores are close to 0, i.e. this "
        "gateway looks like the rest of the group this round; small "
        "nonzero values here are normal noise, not evidence of an "
        "attack).\n"
        "Example 2 (borderline, worth a downweight but not exclusion): "
        "update_norm_z=+1.6, cosine_z=-1.4, calibration loss: got "
        "slightly worse -> anomaly_score=0.4 (this gateway deviates "
        "somewhat from the rest of the group this round, but not "
        "drastically, and the calibration loss change is small).\n"
        "Example 3 (clearly anomalous): update_norm_z=+3.8, "
        "cosine_z=-3.5, calibration loss: got much worse -> "
        "anomaly_score=0.95 (this gateway is a strong outlier vs. the "
        "rest of the group THIS round on both z-scores, and calibration "
        "loss got substantially worse -- multiple independent signals "
        "agree this looks poisoned, not just relatively different from "
        "one fixed absolute number).\n\n"
        "Now judge the real round below the same way -- always relative "
        "to this round's own group, never against a fixed absolute "
        "cosine or z-score value from a different round. Respond with "
        'ONLY a JSON object of the exact form {"anomaly_score": <number '
        "between 0.0 (looks completely normal) and 1.0 (looks highly "
        'anomalous)>, "rationale": "<one sentence naming the specific '
        'digest value(s) that drove the score>"}. No text before or '
        "after the JSON."
    )

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        timeout_s: float = 30.0,
        max_retries: int = 2,
        exclude_thresh: float = 0.7,
        downweight_thresh: float = 0.35,
        long_term_n_obs: float = 5.0,
        long_term_posterior: float = 0.7,
        calibrated: bool = False,
        warmup_rounds: int = 0,
        warmup_scale: float = 1.5,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_retries = max_retries
        self.calibrated = calibrated
        # Section VII-E fix #2, applied here as well since this is the
        # policy the diagnosed D=10 hybrid failure used: widen thresholds
        # (require a higher LLM anomaly_score before exclude/downweight)
        # for the first `warmup_rounds` rounds.
        self.warmup_rounds = warmup_rounds
        self.warmup_scale = warmup_scale
        # Deterministic decision thresholds -- same structure and same
        # numeric values as RuleBasedGatingPolicy's norm-z-score
        # thresholds and long-term-override condition, just applied to
        # the LLM's [0,1] anomaly_score instead of a z-score. Not tuned
        # separately from the rule-based policy's thresholds.
        self.exclude_thresh = exclude_thresh
        self.downweight_thresh = downweight_thresh
        self.long_term_n_obs = long_term_n_obs
        self.long_term_posterior = long_term_posterior

    def _build_prompt(
        self,
        digest: Digest,
        round_update_norms: List[float],
        round_idx: int = -1,
        round_cosines: Optional[List[float]] = None,
    ) -> str:
        norms = np.array(round_update_norms)
        mu, sigma = norms.mean(), norms.std() + 1e-8
        z = (digest.update_norm - mu) / sigma

        cos_z = cosine_zscore(digest.cosine_sim_to_prev_update, round_cosines)
        if cos_z is not None:
            cos_line = f"  cosine similarity z-score vs. {len(round_cosines)} gateways this round: {cos_z:+.2f}\n"
        else:
            cos_line = f"  cosine similarity to previous global update = {digest.cosine_sim_to_prev_update:.3f}\n"

        round_line = f"Training round: {round_idx}\n" if round_idx >= 0 else ""
        return (
            f"{round_line}"
            f"This round's digest for gateway {digest.client_id}:\n"
            f"  update norm z-score vs. {len(round_update_norms)} gateways this round: {z:+.2f}\n"
            f"{cos_line}"
            f"  calibration loss: {describe_calib_delta(digest.calib_loss_delta)}\n"
            f"How anomalous does this round's digest look?"
        )

    def _call_ollama(self, prompt: str) -> dict:
        import json as _json
        import urllib.request

        system_prompt = self.SYSTEM_PROMPT_CALIBRATED if self.calibrated else self.SYSTEM_PROMPT
        payload = _json.dumps({
            "model": self.model,
            "system": system_prompt,
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
            raise LLMGatingPolicyError(
                f"Could not reach Ollama at {self.base_url} (is `ollama serve` "
                f"running, and has `ollama pull {self.model}` been run?): {e}"
            ) from e

        raw_text = body.get("response", "")
        try:
            return _json.loads(raw_text)
        except _json.JSONDecodeError as e:
            raise LLMGatingPolicyError(
                f"Model response was not valid JSON: {raw_text!r}"
            ) from e

    def decide(
        self,
        digest: Digest,
        round_update_norms: List[float],
        trust: "BayesianTrust",
        round_idx: int = -1,
        round_cosines: Optional[List[float]] = None,
        round_calib_deltas: Optional[List[float]] = None,  # unused here; accepted for a uniform policy interface
    ) -> GatingDecision:
        prompt = self._build_prompt(digest, round_update_norms, round_idx=round_idx, round_cosines=round_cosines)
        # Long-term state read BEFORE this round's decision, exactly as
        # RuleBasedGatingPolicy does -- computed and applied entirely
        # outside the LLM call above.
        posterior_mean = trust.mean(digest.client_id, digest.component)
        n_obs = trust.n_observations(digest.client_id, digest.component)
        long_term_confident = n_obs >= self.long_term_n_obs and posterior_mean >= self.long_term_posterior

        scale = early_round_leniency_scale(round_idx, self.warmup_rounds, self.warmup_scale) \
            if round_idx >= 0 else 1.0
        exclude_thresh = min(1.0, self.exclude_thresh * scale)
        downweight_thresh = min(1.0, self.downweight_thresh * scale)

        last_error: Exception = LLMGatingPolicyError("no attempts made")
        for attempt in range(self.max_retries + 1):
            try:
                out = self._call_ollama(prompt)
                anomaly = float(out["anomaly_score"])
                rationale = str(out.get("rationale", ""))
                if not (0.0 <= anomaly <= 1.0):
                    raise LLMGatingPolicyError(f"anomaly_score out of range: {anomaly}")
                if scale != 1.0:
                    rationale += f" (early-round warmup active, scale={scale:.2f})"

                # Deterministic decision logic, identical in structure to
                # RuleBasedGatingPolicy.decide -- only the anomaly signal
                # driving it (LLM judgment vs. fixed z-score) differs.
                if anomaly >= exclude_thresh:
                    if long_term_confident:
                        score, action = 0.5, "downweight"
                        rationale += (
                            f" (long-term posterior mean={posterior_mean:.2f} over "
                            f"{n_obs:.0f} rounds downgrades exclude to downweight)"
                        )
                    else:
                        score, action = 0.0, "exclude"
                elif anomaly >= downweight_thresh:
                    score, action = 0.5, "downweight"
                else:
                    score, action = 1.0, "accept"

                trust.update(digest.client_id, digest.component, score)
                return GatingDecision(digest.client_id, digest.component, score, action, rationale)
            except (LLMGatingPolicyError, KeyError, ValueError, TypeError) as e:
                last_error = e
        raise LLMGatingPolicyError(
            f"Failed to get a valid decision after {self.max_retries + 1} attempts: {last_error}"
        )


class OpenAIExternalBayesianPolicy(LLMExternalBayesianPolicy):
    """Same hybrid architecture and prompt as `LLMExternalBayesianPolicy`
    (identical `_build_prompt`, identical deterministic decision logic in
    `decide`, identical external Bayesian trust posterior), but backed by
    a real hosted frontier model via the OpenAI API instead of a local
    Ollama model -- closes the "LLM pool is two local open-weight models,
    not a hosted frontier model" limitation with real data rather than
    leaving it as an untested gap. Only `_call_ollama` (the backend call)
    is overridden; everything downstream is unchanged, so any accuracy
    difference is attributable to the model itself, not the surrounding
    architecture.
    """

    def __init__(self, *args, api_key: Optional[str] = None, **kwargs):
        super().__init__(*args, **kwargs)
        import os as _os
        self.api_key = api_key or _os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise LLMGatingPolicyError(
                "OPENAI_API_KEY not set and no api_key passed -- required for OpenAIExternalBayesianPolicy"
            )

    def _call_ollama(self, prompt: str) -> dict:
        import json as _json
        import urllib.request

        system_prompt = self.SYSTEM_PROMPT_CALIBRATED if self.calibrated else self.SYSTEM_PROMPT
        payload = _json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }).encode("utf-8")
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                body = _json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            raise LLMGatingPolicyError(f"Could not reach OpenAI API: {e}") from e

        try:
            raw_text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as e:
            raise LLMGatingPolicyError(f"Unexpected OpenAI response shape: {body!r}") from e

        try:
            return _json.loads(raw_text)
        except _json.JSONDecodeError as e:
            raise LLMGatingPolicyError(f"Model response was not valid JSON: {raw_text!r}") from e


class SelectiveLLMGatingPolicy(LLMExternalBayesianPolicy):
    """Cost-reduction variant, motivated directly by the honest finding
    that a fast non-LLM baseline (GShield-style clustering) matches the
    full-LLM Gating Agent's accuracy at ~50-100x lower wall-clock cost
    (Section VII of the paper): if the LLM's real value is the rationale
    attached to a decision, not raw accuracy, then it should only be
    called where a rationale is actually decision-relevant -- i.e. where
    the cheap deterministic signal is NOT already confident the client
    looks normal. A clearly-normal round doesn't need an LLM to say so.

    This is deliberately NOT a copy of GShield's PCA/clustering
    mechanism: it reuses the same digest z-scores and fixed thresholds
    already computed for `RuleBasedGatingPolicy`, applied as a cheap
    admission filter rather than as the final decision. Every client
    whose digest is comfortably inside the "normal" band (well below the
    downweight threshold on ALL three signals, not just one) is accepted
    deterministically, for free. Every other client -- borderline,
    downweight-range, or exclude-range -- gets the real LLM call for its
    anomaly judgment and rationale, using the identical hybrid
    architecture (`LLMExternalBayesianPolicy`) already validated: the
    LLM only ever judges this round's digest in isolation, and the
    Bayesian trust posterior + final decision stay external and
    deterministic.

    `margin` (< 1.0) controls how much of the downweight threshold's
    margin is reserved as a safety buffer before the fast path is
    trusted: a client at 0.9x the downweight z-threshold is still routed
    to the LLM, not accepted for free, since it's close enough to the
    boundary that a rationale is worth having.
    """

    def __init__(
        self,
        *args,
        norm_downweight_z: float = 1.5,
        cosine_safe_z: float = -1.0,
        calib_safe_delta: float = 0.15,
        margin: float = 0.7,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.norm_downweight_z = norm_downweight_z
        self.cosine_safe_z = cosine_safe_z
        self.calib_safe_delta = calib_safe_delta
        self.margin = margin
        # Reporting counters (Section VII: what fraction of rounds
        # actually needed an LLM call).
        self.n_fast_path = 0
        self.n_llm_calls = 0

    def _is_confidently_normal(
        self,
        digest: Digest,
        round_update_norms: List[float],
        round_cosines: Optional[List[float]],
    ) -> bool:
        norms = np.array(round_update_norms)
        mu, sigma = norms.mean(), norms.std() + 1e-8
        z = (digest.update_norm - mu) / sigma
        if z > self.norm_downweight_z * self.margin:
            return False

        cos_z = cosine_zscore(digest.cosine_sim_to_prev_update, round_cosines)
        if cos_z is not None and cos_z < self.cosine_safe_z:
            return False

        if digest.calib_loss_delta > self.calib_safe_delta:
            return False

        return True

    def decide(
        self,
        digest: Digest,
        round_update_norms: List[float],
        trust: "BayesianTrust",
        round_idx: int = -1,
        round_cosines: Optional[List[float]] = None,
        round_calib_deltas: Optional[List[float]] = None,  # unused here; accepted for a uniform policy interface
    ) -> GatingDecision:
        if self._is_confidently_normal(digest, round_update_norms, round_cosines):
            self.n_fast_path += 1
            # Deliberately do NOT update the long-term trust posterior here.
            # Diagnosed real failure (CIC IoT-DIAD trust_building D=20,
            # F1 collapsed to 0.1869 vs. 0.898 for the full-LLM policy):
            # the fast path's noiseless, always-exactly-1.0 score inflates a
            # patient attacker's accumulated trust faster than the real
            # LLM's naturally noisier judgments do (which occasionally
            # register mild anomaly even on an honest round), so by the
            # time the attack starts, `long_term_confident` fires on every
            # round and permanently downgrades exclude to downweight --
            # the attacker is never fully excluded, only ever halved. The
            # long-term prior must be built ONLY from genuine LLM
            # judgments, or a cheap, unaudited shortcut ends up granting
            # more benefit of the doubt than a real judgment would have.
            return GatingDecision(
                digest.client_id, digest.component, 1.0, "accept",
                "fast path: update norm, cosine, and calibration loss all comfortably "
                "within the normal band this round -- no LLM call made, and this round "
                "does not count toward the long-term trust posterior",
            )
        self.n_llm_calls += 1
        return super().decide(digest, round_update_norms, trust, round_idx=round_idx, round_cosines=round_cosines)


class GracefulDegradationPolicy:
    """Explicit, opt-in production wrapper (Section VII limitation:
    "no graceful degradation if the LLM becomes unreachable"). Any
    caller that wants the experimental honesty of `LLMGatingPolicyError`
    propagating uncaught (so a run is never silently mislabeled as
    LLM-gated when the backend was actually down) uses the wrapped
    policy directly, as every result reported in this paper does. A
    caller that wants a real deployment's availability guarantee instead
    wraps it in this class: on any `LLMGatingPolicyError` from the
    primary policy, falls back to the deterministic `RuleBasedGatingPolicy`
    for that one decision, logs the degradation (never silent), and lets
    training continue rather than crashing the round loop.
    """

    def __init__(self, primary_policy, fallback_policy=None):
        self.primary = primary_policy
        self.fallback = fallback_policy or RuleBasedGatingPolicy()
        self.n_degraded = 0
        self.degraded_rounds: List[int] = []

    def decide(
        self,
        digest: Digest,
        round_update_norms: List[float],
        trust: "BayesianTrust",
        round_idx: int = -1,
        round_cosines: Optional[List[float]] = None,
        round_calib_deltas: Optional[List[float]] = None,
    ) -> GatingDecision:
        try:
            return self.primary.decide(
                digest, round_update_norms, trust, round_idx=round_idx,
                round_cosines=round_cosines, round_calib_deltas=round_calib_deltas,
            )
        except LLMGatingPolicyError as e:
            self.n_degraded += 1
            self.degraded_rounds.append(round_idx)
            decision = self.fallback.decide(
                digest, round_update_norms, trust, round_idx=round_idx,
                round_cosines=round_cosines, round_calib_deltas=round_calib_deltas,
            )
            decision.rationale = (
                f"[DEGRADED: LLM backend unreachable ({e}); used deterministic "
                f"rule-based fallback for this decision] {decision.rationale}"
            )
            return decision
