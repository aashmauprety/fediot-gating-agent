"""Attack simulations matching the Threat Model (Section IV) and the
Experimentation protocols (Section VII-D/E): untargeted, targeted, and
trust-building (delayed) attacks on federated updates.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch


def label_flip_untargeted(y: torch.Tensor, num_classes: int, rng: np.random.RandomState) -> torch.Tensor:
    """Untargeted (availability) attack: random label flipping."""
    y_np = y.clone().numpy()
    flips = rng.randint(0, num_classes, size=len(y_np))
    return torch.tensor(flips, dtype=y.dtype)


def label_flip_targeted(
    y: torch.Tensor, source_class: int, target_class: int
) -> torch.Tensor:
    """Targeted (integrity) attack: relabel all `source_class` samples as
    `target_class` (e.g. a specific attack type mislabeled as benign)."""
    y_new = y.clone()
    y_new[y_new == source_class] = target_class
    return y_new


def scale_update(state_dict: Dict[str, torch.Tensor], factor: float) -> Dict[str, torch.Tensor]:
    """Model-poisoning attack: scale the local update by `factor`
    (>1 amplifies the malicious client's influence under uniform FedAvg)."""
    return {k: v * factor for k, v in state_dict.items()}


def blend_toward_reference(
    delta: Dict[str, torch.Tensor], reference_delta: Dict[str, torch.Tensor], lam: float
) -> Dict[str, torch.Tensor]:
    """Adaptive/evasive attack (Section VII-F), constrain-and-scale style
    (cf. Bagdasaryan et al.'s backdoor attack, already cited in this
    paper's threat model): blend a poisoned update's direction toward a
    reference direction, `lam` in [0,1] controlling the blend (0 = the
    naive attack used everywhere else in this paper; 1 = entirely the
    reference direction, diluting the attack's effect to near-zero but
    maximizing stealth). `reference_delta` should be something the
    attacker can plausibly compute -- e.g. the SAME client's own
    clean-label local-training delta (used in `experiment.py`, since
    blending every malicious client toward one shared reference makes
    them a tighter, more mutually-similar clique rather than dispersing
    them). The blended direction is renormalized to the ORIGINAL delta's
    norm, so this only changes direction, not magnitude -- magnitude
    scaling (`scale_update`) is a separate, independent knob."""
    keys = list(delta.keys())
    flat_delta = torch.cat([delta[k].flatten() for k in keys]).float()
    flat_ref = torch.cat([reference_delta[k].flatten() for k in keys]).float()
    orig_norm = flat_delta.norm()
    ref_norm = flat_ref.norm()
    if orig_norm < 1e-12 or ref_norm < 1e-12:
        return delta
    delta_unit = flat_delta / orig_norm
    ref_unit = flat_ref / ref_norm
    blended_unit = (1 - lam) * delta_unit + lam * ref_unit
    blended_norm = blended_unit.norm()
    if blended_norm < 1e-12:
        return delta
    blended_flat = (blended_unit / blended_norm) * orig_norm
    out = {}
    idx = 0
    for k in keys:
        n = delta[k].numel()
        out[k] = blended_flat[idx : idx + n].view_as(delta[k]).to(delta[k].dtype)
        idx += n
    return out


@dataclass
class TrustBuildingSchedule:
    """Section IV / VII-E: a compromised client behaves honestly for
    `delay_rounds` rounds (to inflate its Bayesian trust posterior), then
    switches to the given base attack for all subsequent rounds."""

    delay_rounds: int

    def is_attacking(self, round_idx: int) -> bool:
        return round_idx >= self.delay_rounds
