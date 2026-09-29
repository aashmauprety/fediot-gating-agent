"""Generic federated learning utilities: local training and weighted
aggregation ("FedAvg" when weights are uniform, "gate-weighted FedAvg" per
Section VI-B when weights come from the Gating Agent).
"""
from __future__ import annotations

import copy
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn


def get_state(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def set_state(model: nn.Module, state: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state)


def state_diff_norm(state_a: Dict[str, torch.Tensor], state_b: Dict[str, torch.Tensor]) -> float:
    total = 0.0
    for k in state_a:
        total += (state_a[k].float() - state_b[k].float()).pow(2).sum().item()
    return total ** 0.5


def state_cosine_similarity(
    delta_a: Dict[str, torch.Tensor], delta_b: Dict[str, torch.Tensor]
) -> float:
    flat_a = torch.cat([v.float().flatten() for v in delta_a.values()])
    flat_b = torch.cat([v.float().flatten() for v in delta_b.values()])
    denom = (flat_a.norm() * flat_b.norm()).clamp(min=1e-12)
    return (flat_a @ flat_b / denom).item()


def state_delta(new: Dict[str, torch.Tensor], old: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: new[k].float() - old[k].float() for k in new}


def weighted_average_states(
    states: Sequence[Dict[str, torch.Tensor]], weights: Sequence[float]
) -> Dict[str, torch.Tensor]:
    """Weighted average of state dicts (plain FedAvg if weights are equal,
    gate-weighted aggregation, Eq. in Section VI-B1, if not)."""
    total_w = sum(weights)
    assert total_w > 0, "all clients excluded this round -- nothing to aggregate"
    keys = states[0].keys()
    out = {}
    for k in keys:
        acc = torch.zeros_like(states[0][k], dtype=torch.float32)
        for st, w in zip(states, weights):
            acc += st[k].float() * (w / total_w)
        out[k] = acc.to(states[0][k].dtype)
    return out


def trimmed_mean_aggregate(
    states: Sequence[Dict[str, torch.Tensor]], trim_frac: float = 0.2
) -> Dict[str, torch.Tensor]:
    """Coordinate-wise trimmed mean (Yin et al. 2018), a robust-aggregation
    baseline used for comparison in Section VII-D."""
    n = len(states)
    k = int(n * trim_frac)
    keys = states[0].keys()
    out = {}
    for key in keys:
        stacked = torch.stack([st[key].float() for st in states], dim=0)  # (n, ...)
        sorted_vals, _ = torch.sort(stacked, dim=0)
        if k > 0 and n - 2 * k > 0:
            trimmed = sorted_vals[k : n - k]
        else:
            trimmed = sorted_vals
        out[key] = trimmed.mean(dim=0).to(states[0][key].dtype)
    return out


def krum_aggregate(
    states: Sequence[Dict[str, torch.Tensor]], num_malicious_assumed: int
) -> Dict[str, torch.Tensor]:
    """Krum (Blanchard et al. 2017): pick the single update whose sum of
    squared distances to its n-f-2 nearest neighbors is smallest."""
    n = len(states)
    flat = [torch.cat([v.float().flatten() for v in st.values()]) for st in states]
    dists = torch.zeros(n, n)
    for i in range(n):
        for j in range(n):
            if i != j:
                dists[i, j] = (flat[i] - flat[j]).pow(2).sum()
    f = min(num_malicious_assumed, max(0, (n - 3) // 2))
    k = max(1, n - f - 2)
    scores = []
    for i in range(n):
        nearest = torch.topk(dists[i], k, largest=False).values
        scores.append(nearest.sum().item())
    best = scores.index(min(scores))
    return states[best]


def gshield_aggregate(
    states: Sequence[Dict[str, torch.Tensor]],
    n_components: int = 10,
    z_thresh: float = 3.0,
) -> Dict[str, torch.Tensor]:
    """Non-LLM, current-generation (2025-era) robust-aggregation baseline,
    inspired by GShield (Sameera et al., arXiv 2512.19286, Dec 2025): learn
    the distribution of benign gradients via clustering + Gaussian
    modeling, then exclude updates that don't fit it. This is our own
    lightweight instantiation of that idea, not a literal reproduction of
    their published method (their exact feature space and clustering
    procedure are not fully specified in the abstract this was built
    from) -- flattened per-client updates are PCA-reduced to
    `n_components` dimensions (raw parameter-space covariance is
    ill-conditioned with only a handful of clients per round), 2-means
    clustered, the larger cluster is treated as the benign baseline, and
    a Gaussian is fit to it; any client (in either cluster) whose
    Mahalanobis distance to that Gaussian exceeds `z_thresh` standard
    deviations of the benign cluster's own distances is excluded from
    the aggregate. Unlike Krum, this requires no assumption about the
    number of malicious clients.
    """
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    n = len(states)
    flat = np.stack(
        [torch.cat([v.float().flatten() for v in st.values()]).numpy() for st in states]
    )
    k = min(n_components, n - 1, flat.shape[1])
    if k < 1:
        return weighted_average_states(states, [1.0] * n)
    reduced = PCA(n_components=k, random_state=0).fit_transform(flat)

    if n >= 4:
        labels = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(reduced)
        counts = np.bincount(labels)
        benign_label = int(np.argmax(counts))
        benign_mask = labels == benign_label
        if benign_mask.sum() < 2:
            benign_mask = np.ones(n, dtype=bool)
    else:
        benign_mask = np.ones(n, dtype=bool)

    benign_pts = reduced[benign_mask]
    mean = benign_pts.mean(axis=0)
    cov = np.cov(benign_pts, rowvar=False) + 1e-6 * np.eye(k)
    inv_cov = np.linalg.pinv(cov)

    def mahalanobis(x):
        d = x - mean
        return float(np.sqrt(d @ inv_cov @ d.T))

    benign_dists = np.array([mahalanobis(x) for x in benign_pts])
    benign_mu, benign_sigma = benign_dists.mean(), benign_dists.std() + 1e-8

    weights = []
    for i in range(n):
        dist = mahalanobis(reduced[i])
        z = (dist - benign_mu) / benign_sigma
        weights.append(0.0 if z > z_thresh else 1.0)
    if sum(weights) == 0:
        weights = [1.0] * n  # safety: never fully stall training
    return weighted_average_states(states, weights)


def local_train_step(
    model: nn.Module,
    loss_fn,
    optimizer_cls,
    lr: float,
    epochs: int,
    data_iter_fn,
    device: str = "cpu",
) -> nn.Module:
    """Run local training on a copy of `model` for `epochs` epochs.
    `data_iter_fn()` must return an iterable of (x, y) batches.
    Returns the locally-trained model (a fresh copy, caller decides what
    to do with its state_dict)."""
    local_model = copy.deepcopy(model).to(device)
    local_model.train()
    opt = optimizer_cls(local_model.parameters(), lr=lr)
    for _ in range(epochs):
        for x, y in data_iter_fn():
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = local_model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
    return local_model


def batches(X: torch.Tensor, y: torch.Tensor, batch_size: int, shuffle: bool = True):
    n = X.shape[0]
    idx = torch.randperm(n) if shuffle else torch.arange(n)
    for i in range(0, n, batch_size):
        b = idx[i : i + batch_size]
        yield X[b], y[b]
