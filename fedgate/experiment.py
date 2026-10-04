"""End-to-end federated simulation: dual-purpose script used for
Section VII-C (attack-type head validation), VII-D (robustness under
attack), and VII-E (trust-building attack + Bayesian prior).

Runs on REAL N-BaIoT data (9 devices = 9 federated clients). The
attack-type head trains directly on N-BaIoT's 115 flow features (no
Phase 0 encoder involved -- see data.py docstring for why). The Gating
Agent's decision policy is the RuleBasedGatingPolicy stand-in documented
in gating_agent.py, not the paper's LLM-based agent.
"""
from __future__ import annotations

import copy
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score

from . import federated as fed
from .attacks import (
    label_flip_untargeted, label_flip_targeted, label_flip_partial, scale_update, blend_toward_reference,
    TrustBuildingSchedule,
)
from .data import ClientData, train_val_test_split
from .gating_agent import BayesianTrust, Digest, GatingDecision, RuleBasedGatingPolicy
from .models import ClassifierHead


@dataclass
class RoundLog:
    round_idx: int
    macro_f1: float
    targeted_success_rate: Optional[float] = None
    actions: Dict[str, str] = field(default_factory=dict)
    gating_scores: Dict[str, float] = field(default_factory=dict)


class FederatedSimulation:
    def __init__(
        self,
        clients: Dict[str, ClientData],
        num_classes: int,
        embed_dim: int,
        malicious_clients: List[str],
        attack: str = "none",  # "none" | "untargeted" | "targeted" | "trust_building"
        gating: bool = False,
        aggregation: str = "fedavg",  # "fedavg" | "trimmed_mean" | "krum" | "gated" (gated implied by gating=True)
        targeted_source: int = 1,
        targeted_target: int = 0,
        trust_building_delay: int = 0,
        lr: float = 1e-3,
        local_epochs: int = 1,
        batch_size: int = 128,
        device: str = "cpu",
        seed: int = 0,
        scale_factor: float = 5.0,
        use_bayesian_prior: bool = True,
        gating_policy=None,
        trust_decay: float = 1.0,
        force_accept_round0: bool = True,
        evasion_lambda: float = 0.0,
        evasion_collude: bool = False,
        evasion_calib_aware: bool = False,
        calib_aware_safe_delta: float = 0.5,
        calib_aware_lambda_grid: Optional[List[float]] = None,
        concept_drift_start_round: int = -1,
        concept_drift_per_round: float = 0.0,
        slow_drip_ramp_rounds: int = 20,
        intermittent_period: int = 3,
    ):
        self.device_str = device
        self.rng = np.random.RandomState(seed)
        torch.manual_seed(seed)

        self.client_ids = list(clients.keys())
        self.splits = {cid: train_val_test_split(c, seed=seed) for cid, c in clients.items()}

        # Global feature scaling from all clients' training data. N-BaIoT's
        # flow-statistic features are extremely heavy-tailed (some variance
        # features reach ~1e17), so plain mean/std in float32 overflows;
        # we compute robust statistics (median / IQR) in float64 instead.
        all_train_X = np.concatenate(
            [self.splits[c][0].X for c in self.client_ids], axis=0
        ).astype(np.float64)
        median = np.median(all_train_X, axis=0)
        q75, q25 = np.percentile(all_train_X, [75, 25], axis=0)
        iqr = (q75 - q25)
        iqr[iqr < 1e-6] = 1.0
        self.mu = median
        self.sigma = iqr

        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.global_model = ClassifierHead(embed_dim, num_classes).to(device)

        # Held-out calibration set for real calib_loss_delta (Section
        # VII-E fix: this used to be hardcoded to 0.0 in every Digest,
        # which the LLM Gating Agent sometimes still misread as evidence
        # of an anomaly -- see README "Ninth real result"). Built from
        # each client's val split (never used for local training or for
        # _eval_global's test-set metric), concatenated once so every
        # round's delta is measured against the same fixed reference set.
        val_X = np.concatenate([self.splits[c][1].X for c in self.client_ids], axis=0)
        val_y = np.concatenate([self.splits[c][1].y_attack for c in self.client_ids], axis=0)
        self._calib_X = self._standardize(val_X).to(device)
        self._calib_y = torch.tensor(val_y, dtype=torch.long).to(device)
        self._calib_loss_fn = nn.CrossEntropyLoss()

        self.malicious = set(malicious_clients)
        self.attack = attack
        self.gating = gating
        self.aggregation = aggregation
        self.targeted_source = targeted_source
        self.targeted_target = targeted_target
        self.trust_schedule = TrustBuildingSchedule(trust_building_delay)
        self.scale_factor = scale_factor
        self.evasion_lambda = evasion_lambda
        # Colluding-attacker variant (Section VII-F follow-up): malicious
        # clients pool their own clean-label gradients into a single
        # shared reference for the round, instead of each blending toward
        # its OWN clean gradient in isolation -- a stronger threat model
        # (requires malicious clients to coordinate/share local data
        # statistics with each other) that tests whether a more
        # representative, less-noisy reference direction evades detection
        # more effectively than each client's own noisier local estimate.
        self.evasion_collude = evasion_collude
        self._collude_ref_round = -1
        self._collude_ref_delta: Optional[Dict[str, torch.Tensor]] = None

        # Calibration-preserving adaptive attacker (review-response
        # follow-up, Oct 2026): every evasion variant above was built to
        # evade the update-norm/cosine digest signals only, never tested
        # against calib_loss_delta at all -- an attacker could in
        # principle pick a blend that looks directionally honest (fools
        # norm/cosine) while still making the global model measurably
        # worse on held-out data. This variant searches a grid of blend
        # strengths (floor at `evasion_lambda`, same reference-direction
        # mechanism as above) and picks the smallest (= strongest
        # residual attack) one whose calibration impact -- measured on
        # the attacker's OWN held-out val split, a conservative,
        # genuinely-available local proxy for the server's real
        # calib_loss_delta, since the server's calibration set is built
        # by concatenating every client's val split including this one
        # -- stays at or below `calib_aware_safe_delta`. This assumes the
        # attacker knows that threshold value, the same white-box
        # worst-case assumption the constrain-and-scale literature
        # already makes for norm/cosine (Bagdasaryan et al., already
        # cited). It does NOT assume the attacker can see other clients'
        # calib deltas this round, so it cannot directly target the real
        # policy's same-round cross-client z-score -- only its own
        # absolute impact, a realistic (not omniscient) attacker.
        self.evasion_calib_aware = evasion_calib_aware
        self.calib_aware_safe_delta = calib_aware_safe_delta
        self.calib_aware_lambda_grid = calib_aware_lambda_grid or [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

        # Synthetic benign concept drift (Section VII limitation: "no
        # concept-drift adaptation"): a slowly-growing additive bias
        # applied to EVERY client's calib_loss_delta (not just malicious
        # ones) starting at `concept_drift_start_round`, representing a
        # genuine, non-adversarial shift in the calibration set's natural
        # loss over a long real deployment (e.g. the benign traffic
        # distribution itself changing), independent of any attack. Used
        # only in the drift-adaptation experiment, never in the poisoning
        # results reported elsewhere in the paper (default disabled).
        self.concept_drift_start_round = concept_drift_start_round
        self.concept_drift_per_round = concept_drift_per_round

        # Novel-attack-shape generalization test (Section VII limitation
        # follow-up): none of these three shapes appear in the Gating
        # Agent's few-shot calibration examples, which were hand-built
        # against untargeted/targeted/trust-building digest signatures --
        # testing them WITHOUT touching the prompt checks whether the
        # design generalizes or was implicitly overfit to those examples.
        #   "slow_drip": attack strength (both label-flip fraction and
        #     magnitude scaling) ramps linearly from 0 to full strength
        #     over `slow_drip_ramp_rounds`, instead of switching on at
        #     full strength immediately -- each individual round looks
        #     only mildly anomalous.
        #   "free_rider": a malicious client submits no real update at
        #     all (returns the global model essentially unchanged) --
        #     not poisoning, a distinct, real FL threat category (a
        #     client shirking compute/data contribution while still
        #     being counted and rewarded).
        #   "intermittent": attacks only every `intermittent_period`-th
        #     round, honest otherwise -- tests whether the long-term
        #     trust posterior handles a recurring pattern rather than a
        #     one-time sustained shift.
        self.slow_drip_ramp_rounds = slow_drip_ramp_rounds
        self.intermittent_period = intermittent_period

        self.lr = lr
        self.local_epochs = local_epochs
        self.batch_size = batch_size

        # gating_policy lets a caller swap in LLMGatingPolicy (or any object
        # with a matching .decide(digest, round_norms, trust) method)
        # instead of the rule-based stand-in, without touching any of the
        # rule-based results already collected.
        self.policy = gating_policy if gating_policy is not None else RuleBasedGatingPolicy(use_bayesian_prior=use_bayesian_prior)
        self.trust = BayesianTrust(decay=trust_decay)
        self.prev_global_delta: Optional[Dict[str, torch.Tensor]] = None
        # Section VII-E fix #1: round 0 has no previous global update, so
        # cosine_sim_to_prev_update is hardcoded to 1.0 below purely
        # because there is nothing real to compare against -- not because
        # the client's update is actually normal. Left to the policy,
        # this makes round 0 trivially easy for every client and looked
        # like "calibration working" when it was a code artifact. With
        # `force_accept_round0=True` (default), round 0 is accepted for
        # every client without consulting the policy or updating the
        # trust posterior, so it can never masquerade as real evidence
        # either way.
        self.force_accept_round0 = force_accept_round0

        self.logs: List[RoundLog] = []
        # Per-round gating decisions (action, score, rationale), recorded
        # EVERY round regardless of log_every -- unlike self.logs (which
        # only stores expensive macro-F1 eval at log_every intervals),
        # this is cheap (gating already runs every round) and was added
        # specifically because its absence blocked diagnosing a real,
        # unexplained result (Section VII-E's calibration-fix regression:
        # only round-0 and final-round snapshots existed, insufficient to
        # tell whether behavior changed mid-run). See gating_history.
        self.gating_history: List[dict] = []

    def _calib_loss(self, state: Dict[str, torch.Tensor]) -> float:
        """Held-out cross-entropy loss of `state` on the fixed calibration
        set -- used to compute each client's real calib_loss_delta."""
        model = copy.deepcopy(self.global_model)
        fed.set_state(model, state)
        model.eval()
        with torch.no_grad():
            logits = model(self._calib_X)
            loss = self._calib_loss_fn(logits, self._calib_y)
        return float(loss.item())

    def _local_calib_proxy_loss(self, state: Dict[str, torch.Tensor], cid: str) -> float:
        """Loss of `state` on client `cid`'s OWN val split -- a local
        proxy for the server's real calib_loss_delta, used only by the
        calibration-preserving adaptive attacker (`evasion_calib_aware`)
        to estimate its own detectability without needing any
        cross-client information. Genuinely available to the client:
        it's the same val split `train_val_test_split` already set
        aside, never used for local training."""
        _, val, _ = self.splits[cid]
        X = self._standardize(val.X).to(self.device_str)
        y = torch.tensor(val.y_attack, dtype=torch.long).to(self.device_str)
        model = copy.deepcopy(self.global_model)
        fed.set_state(model, state)
        model.eval()
        with torch.no_grad():
            logits = model(X)
            loss = self._calib_loss_fn(logits, y)
        return float(loss.item())

    def _local_calib_proxy_loss_pooled(self, state: Dict[str, torch.Tensor]) -> float:
        """Colluding variant of `_local_calib_proxy_loss`: pools ALL
        malicious clients' val splits into one less-noisy proxy, exactly
        the same realistic benefit collusion already gives the
        direction-blending reference (`_get_collude_reference`) --
        malicious clients sharing local data statistics with each other,
        never with honest clients or the server."""
        Xs, ys = [], []
        for mcid in self.malicious:
            _, val, _ = self.splits[mcid]
            Xs.append(val.X)
            ys.append(val.y_attack)
        import numpy as _np
        X = self._standardize(_np.concatenate(Xs, axis=0)).to(self.device_str)
        y = torch.tensor(_np.concatenate(ys, axis=0), dtype=torch.long).to(self.device_str)
        model = copy.deepcopy(self.global_model)
        fed.set_state(model, state)
        model.eval()
        with torch.no_grad():
            logits = model(X)
            loss = self._calib_loss_fn(logits, y)
        return float(loss.item())

    def _select_calib_aware_lambda(
        self, cid: str, raw_delta: Dict[str, torch.Tensor], reference_delta: Dict[str, torch.Tensor],
        effective_scale: float, prev_state: Dict[str, torch.Tensor], use_pooled_proxy: bool = False,
    ) -> float:
        """Grid-search blend strength to find the strongest residual
        attack (smallest lambda >= the fixed `evasion_lambda` floor)
        whose calibration impact stays within `calib_aware_safe_delta`;
        falls back to whichever candidate minimizes that impact if none
        qualify. `use_pooled_proxy=True` (the colluding variant) judges
        impact against the pooled malicious-client val set instead of
        just `cid`'s own -- a less noisy, but still genuinely
        locally-available (to the colluding cohort), estimate."""
        proxy_fn = self._local_calib_proxy_loss_pooled if use_pooled_proxy else \
            (lambda state: self._local_calib_proxy_loss(state, cid))
        base_proxy_loss = proxy_fn(prev_state)
        best_lambda, best_abs_delta = 1.0, float("inf")
        for lam in self.calib_aware_lambda_grid:
            if lam < self.evasion_lambda:
                continue
            blended = blend_toward_reference(raw_delta, reference_delta, lam)
            blended = scale_update(blended, effective_scale)
            candidate_state = {k: prev_state[k] + blended[k].to(prev_state[k].dtype) for k in prev_state}
            proxy_delta = proxy_fn(candidate_state) - base_proxy_loss
            if proxy_delta <= self.calib_aware_safe_delta:
                return lam
            if abs(proxy_delta) < best_abs_delta:
                best_abs_delta = abs(proxy_delta)
                best_lambda = lam
        return best_lambda

    def _standardize(self, X: np.ndarray) -> torch.Tensor:
        z = (X.astype(np.float64) - self.mu) / self.sigma
        z = np.clip(z, -20.0, 20.0)  # robust-scaled but still heavy-tailed; clip residual outliers
        return torch.tensor(z, dtype=torch.float32)

    def _client_tensors(self, cid: str, round_idx: int):
        train, _, _ = self.splits[cid]
        X = self._standardize(train.X)
        y = torch.tensor(train.y_attack, dtype=torch.long)

        is_attacking = cid in self.malicious
        if self.attack == "trust_building":
            is_attacking = is_attacking and self.trust_schedule.is_attacking(round_idx)
        if self.attack == "intermittent":
            is_attacking = is_attacking and (round_idx % self.intermittent_period == 0)

        if is_attacking and self.attack in ("untargeted", "trust_building", "intermittent"):
            y = label_flip_untargeted(y, self.num_classes, self.rng)
        elif is_attacking and self.attack == "targeted":
            y = label_flip_targeted(y, self.targeted_source, self.targeted_target)
        elif is_attacking and self.attack == "slow_drip":
            fraction = min(1.0, round_idx / max(1, self.slow_drip_ramp_rounds))
            y = label_flip_partial(y, fraction, self.num_classes, self.rng)
        return X, y

    def _local_train(self, cid: str, round_idx: int) -> Dict[str, torch.Tensor]:
        X, y = self._client_tensors(cid, round_idx)

        def data_iter():
            yield from fed.batches(X, y, self.batch_size)

        loss_fn = nn.CrossEntropyLoss()
        local_model = fed.local_train_step(
            self.global_model, loss_fn, torch.optim.Adam, self.lr,
            self.local_epochs, data_iter, device=self.device_str,
        )
        state = fed.get_state(local_model)

        is_attacking = cid in self.malicious
        if self.attack == "trust_building":
            is_attacking = is_attacking and self.trust_schedule.is_attacking(round_idx)
        if self.attack == "intermittent":
            is_attacking = is_attacking and (round_idx % self.intermittent_period == 0)

        if is_attacking and self.attack == "free_rider":
            # Not poisoning -- a distinct real FL threat: submits (almost)
            # no real update at all, shirking its compute/data
            # contribution while still being counted in the average. Tiny
            # random noise instead of an exact zero delta, so downstream
            # cosine-similarity computation never divides by a zero norm.
            prev = fed.get_state(self.global_model)
            return {k: prev[k] + torch.randn_like(prev[k]) * 1e-6 for k in prev}

        if is_attacking and self.attack in ("untargeted", "targeted", "trust_building", "intermittent", "slow_drip"):
            # also scale the update to make the poisoning attempt stronger,
            # not just relabeled local training
            prev = fed.get_state(self.global_model)
            delta = fed.state_delta(state, prev)
            effective_scale = self.scale_factor
            if self.attack == "slow_drip":
                # Ramp magnitude scaling up alongside the label-flip
                # fraction, so each individual round looks only mildly
                # anomalous rather than switching on at full strength.
                fraction = min(1.0, round_idx / max(1, self.slow_drip_ramp_rounds))
                effective_scale = 1.0 + fraction * (self.scale_factor - 1.0)
            if self.evasion_lambda > 0.0:
                # Constrain-and-scale-style adaptive attack (Section
                # VII-F): blend toward THIS client's own clean-label
                # gradient (computable locally, no cross-client
                # information needed), not a shared reference -- blending
                # every malicious client toward the SAME direction (e.g.
                # the previous global update) makes them a tighter, more
                # mutually-similar clique and is easier, not harder, to
                # cluster apart; blending toward each client's own honest
                # direction disperses them the way real honest clients
                # are already dispersed from each other.
                #
                # `evasion_collude=True` tests a stronger, coordinated
                # attacker that pools all malicious clients' clean
                # gradients into one shared, less-noisy reference instead.
                if self.evasion_collude:
                    reference_delta = self._get_collude_reference(round_idx, prev)
                else:
                    train, _, _ = self.splits[cid]
                    X_clean = self._standardize(train.X)
                    y_clean = torch.tensor(train.y_attack, dtype=torch.long)

                    def clean_iter():
                        yield from fed.batches(X_clean, y_clean, self.batch_size)

                    clean_model = fed.local_train_step(
                        self.global_model, nn.CrossEntropyLoss(), torch.optim.Adam,
                        self.lr, self.local_epochs, clean_iter, device=self.device_str,
                    )
                    reference_delta = fed.state_delta(fed.get_state(clean_model), prev)
                lam = self.evasion_lambda
                if self.evasion_calib_aware:
                    lam = self._select_calib_aware_lambda(
                        cid, delta, reference_delta, effective_scale, prev,
                        use_pooled_proxy=self.evasion_collude,
                    )
                delta = blend_toward_reference(delta, reference_delta, lam)
            delta = scale_update(delta, effective_scale)
            state = {k: prev[k] + delta[k].to(prev[k].dtype) for k in prev}
        return state

    def _get_collude_reference(self, round_idx: int, prev: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Average clean-label gradient across ALL malicious clients this
        round, computed once and cached (not once per malicious client) --
        a coordinated attacker's shared reference direction."""
        if self._collude_ref_round == round_idx and self._collude_ref_delta is not None:
            return self._collude_ref_delta
        deltas = []
        for mcid in self.malicious:
            train, _, _ = self.splits[mcid]
            X_clean = self._standardize(train.X)
            y_clean = torch.tensor(train.y_attack, dtype=torch.long)

            def clean_iter():
                yield from fed.batches(X_clean, y_clean, self.batch_size)

            clean_model = fed.local_train_step(
                self.global_model, nn.CrossEntropyLoss(), torch.optim.Adam,
                self.lr, self.local_epochs, clean_iter, device=self.device_str,
            )
            deltas.append(fed.state_delta(fed.get_state(clean_model), prev))
        avg_delta = {k: sum(d[k] for d in deltas) / len(deltas) for k in deltas[0]}
        self._collude_ref_round = round_idx
        self._collude_ref_delta = avg_delta
        return avg_delta

    def _eval_global(self) -> float:
        self.global_model.eval()
        all_preds, all_true = [], []
        with torch.no_grad():
            for cid in self.client_ids:
                _, _, test = self.splits[cid]
                X = self._standardize(test.X).to(self.device_str)
                logits = self.global_model(X)
                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.append(preds)
                all_true.append(test.y_attack)
        all_preds = np.concatenate(all_preds)
        all_true = np.concatenate(all_true)
        return f1_score(all_true, all_preds, average="macro", zero_division=0)

    def _targeted_success_rate(self) -> Optional[float]:
        if self.attack != "targeted" and not (self.attack == "trust_building"):
            return None
        self.global_model.eval()
        preds_source, n_source = 0, 0
        with torch.no_grad():
            for cid in self.client_ids:
                _, _, test = self.splits[cid]
                mask = test.y_attack == self.targeted_source
                if mask.sum() == 0:
                    continue
                X = self._standardize(test.X[mask]).to(self.device_str)
                logits = self.global_model(X)
                preds = logits.argmax(dim=1).cpu().numpy()
                preds_source += (preds == self.targeted_target).sum()
                n_source += mask.sum()
        return float(preds_source) / max(1, n_source)

    def run(self, num_rounds: int, log_every: int = 1) -> List[RoundLog]:
        for r in range(num_rounds):
            client_states = []
            digests = []
            # Loss of the (unchanged, pre-aggregation) global model on the
            # held-out calibration set, shared by every client's delta
            # this round -- only computed when gating actually consumes
            # calib_loss_delta, since it costs an extra forward pass per
            # client otherwise.
            base_state = fed.get_state(self.global_model)
            calib_loss_before = self._calib_loss(base_state) if self.gating else None
            for cid in self.client_ids:
                prev_state = fed.get_state(self.global_model)
                new_state = self._local_train(cid, r)
                delta = fed.state_delta(new_state, prev_state)
                norm = sum(v.pow(2).sum().item() for v in delta.values()) ** 0.5
                cos = 1.0
                if self.prev_global_delta is not None:
                    cos = fed.state_cosine_similarity(delta, self.prev_global_delta)
                calib_loss_delta = 0.0
                if self.gating:
                    calib_loss_delta = self._calib_loss(new_state) - calib_loss_before
                    if self.concept_drift_start_round >= 0 and r >= self.concept_drift_start_round:
                        rounds_into_drift = r - self.concept_drift_start_round + 1
                        calib_loss_delta += rounds_into_drift * self.concept_drift_per_round
                client_states.append(new_state)
                digests.append(Digest(cid, "attack-head", norm, cos, calib_loss_delta))

            actions, scores = {}, {}
            prev_state = fed.get_state(self.global_model)
            if self.gating:
                norms = [d.update_norm for d in digests]
                cosines = [d.cosine_sim_to_prev_update for d in digests]
                calib_deltas = [d.calib_loss_delta for d in digests]
                weights = []
                rationales = {}
                for d in digests:
                    if r == 0 and self.force_accept_round0:
                        decision = GatingDecision(
                            d.client_id, d.component, 1.0, "accept",
                            "round 0: no previous global update exists yet, cosine "
                            "similarity to it is undefined -- trivially accepted "
                            "without consulting the policy or updating the trust "
                            "posterior, not a real gating judgment",
                        )
                    else:
                        decision = self.policy.decide(
                            d, norms, self.trust, round_idx=r, round_cosines=cosines,
                            round_calib_deltas=calib_deltas,
                        )
                    w = decision.gating_score if decision.action != "exclude" else 0.0
                    weights.append(w)
                    actions[d.client_id] = decision.action
                    scores[d.client_id] = decision.gating_score
                    rationales[d.client_id] = decision.rationale
                self.gating_history.append({
                    "round": r, "actions": dict(actions), "scores": dict(scores),
                    "rationales": rationales, "fallback_triggered": sum(weights) == 0,
                })
                if sum(weights) == 0:
                    weights = [1.0] * len(weights)  # safety: never fully stall training
                new_global = fed.weighted_average_states(client_states, weights)
            elif self.aggregation == "trimmed_mean":
                new_global = fed.trimmed_mean_aggregate(client_states, trim_frac=0.2)
            elif self.aggregation == "krum":
                new_global = fed.krum_aggregate(client_states, num_malicious_assumed=len(self.malicious))
            elif self.aggregation == "gshield":
                new_global = fed.gshield_aggregate(client_states)
            else:
                weights = [1.0] * len(client_states)
                new_global = fed.weighted_average_states(client_states, weights)
            self.prev_global_delta = fed.state_delta(new_global, prev_state)
            fed.set_state(self.global_model, new_global)

            if r % log_every == 0 or r == num_rounds - 1:
                f1 = self._eval_global()
                tsr = self._targeted_success_rate()
                self.logs.append(RoundLog(r, f1, tsr, actions, scores))
        return self.logs

    def logs_as_dicts(self) -> List[dict]:
        return [
            {
                "round": l.round_idx,
                "macro_f1": l.macro_f1,
                "targeted_success_rate": l.targeted_success_rate,
            }
            for l in self.logs
        ]
