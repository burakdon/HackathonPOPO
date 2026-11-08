"""Fit (s, p) to a patient's naming response counts.

Dell et al. (2004, 2013): search the (s, p) space for the parameters that maximise the likelihood
of the observed response-category counts under the model's simulated probabilities. We simulate
the six-category probabilities once on a grid, cache the table, and then fitting a patient is a
lookup: the maximum-likelihood cell, the root-mean-squared deviation Dell reports, and (because
the grid makes it free) a posterior over the grid under a uniform prior, as WebFit does.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from patientmodel.dell_sp import NAMING_CATEGORIES, DellSPConfig, DellSPModel

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"


@dataclass
class GridTable:
    s_values: np.ndarray          # (S,)
    p_values: np.ndarray          # (P,)
    probs: np.ndarray             # (S, P, 6) simulated probabilities over NAMING_CATEGORIES
    trials_per_cell: int

    def lookup(self, s: float, p: float) -> np.ndarray:
        i = int(np.abs(self.s_values - s).argmin()); j = int(np.abs(self.p_values - p).argmin())
        return self.probs[i, j]


def build_grid(model: DellSPModel, grid_step: float = 0.002, trials_per_cell: int = 4000,
               seed: int = 0, weight_max: float | None = None) -> GridTable:
    wmax = weight_max if weight_max is not None else model.cfg.weight_max
    values = np.round(np.arange(0.0, wmax + 1e-9, grid_step), 6)
    probs = np.zeros((len(values), len(values), len(NAMING_CATEGORIES)))
    rng = np.random.default_rng(seed)
    for i, s in enumerate(values):
        for j, p in enumerate(values):
            cat = model.simulate_trials(float(s), float(p), trials_per_cell, rng)
            counts = np.bincount(cat, minlength=len(NAMING_CATEGORIES) + 1)[: len(NAMING_CATEGORIES)]
            probs[i, j] = counts / counts.sum()
    return GridTable(values, values, probs, trials_per_cell)


def grid_cache_path(cfg: DellSPConfig, grid_step: float, trials_per_cell: int, seed: int) -> Path:
    key = f"dellsp_d{cfg.decay}_n{cfg.intrinsic_noise_sd}-{cfg.activation_noise_sd}_sh{cfg.shared_features}" \
          f"_max{cfg.weight_max}_step{grid_step}_n{trials_per_cell}_seed{seed}.npz"
    return CACHE_DIR / key


def load_or_build_grid(model: DellSPModel, grid_step: float = 0.002, trials_per_cell: int = 4000,
                       seed: int = 0, cache: bool = True) -> GridTable:
    path = grid_cache_path(model.cfg, grid_step, trials_per_cell, seed)
    if cache and path.exists():
        z = np.load(path)
        return GridTable(z["s_values"], z["p_values"], z["probs"], int(z["trials_per_cell"]))
    table = build_grid(model, grid_step, trials_per_cell, seed)
    if cache:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, s_values=table.s_values, p_values=table.p_values,
                            probs=table.probs, trials_per_cell=table.trials_per_cell)
    return table


@dataclass
class Fit:
    s: float
    p: float
    rmsd: float                     # Dell's fit statistic: RMS deviation between observed and model proportions
    log_likelihood: float
    predicted: dict[str, float]     # model proportions at the ML cell
    posterior_mean: tuple[float, float]
    s_interval: tuple[float, float]  # 95% central posterior interval (grid, uniform prior)
    p_interval: tuple[float, float]


def counts_from_proportions(proportions: dict[str, float], n_items: int = 175) -> dict[str, int]:
    """Turn published proportions (which sum to about 1) into integer counts over n_items."""
    raw = np.array([proportions.get(k, 0.0) for k in NAMING_CATEGORIES], dtype=float)
    raw = raw / raw.sum() * n_items
    counts = np.floor(raw).astype(int)
    for k in np.argsort(raw - counts)[::-1][: n_items - counts.sum()]:
        counts[k] += 1
    return dict(zip(NAMING_CATEGORIES, counts.tolist()))


def fit_counts(counts: dict[str, int], table: GridTable, probability_floor: float = 1e-4) -> Fit:
    """Maximum-likelihood (s, p) for observed category counts, plus the grid posterior."""
    n = np.array([counts.get(k, 0) for k in NAMING_CATEGORIES], dtype=float)
    if n.sum() <= 0:
        raise ValueError("counts are empty")
    probs = np.clip(table.probs, probability_floor, None)
    probs = probs / probs.sum(axis=-1, keepdims=True)
    ll = (n * np.log(probs)).sum(axis=-1)                       # (S, P) multinomial log-likelihood (up to a constant)
    i, j = np.unravel_index(int(ll.argmax()), ll.shape)
    obs = n / n.sum()
    pred = table.probs[i, j]
    rmsd = float(np.sqrt(np.mean((obs - pred) ** 2)))

    post = np.exp(ll - ll.max()); post /= post.sum()
    ps = post.sum(axis=1); pp = post.sum(axis=0)
    s_mean = float((ps * table.s_values).sum()); p_mean = float((pp * table.p_values).sum())
    return Fit(
        s=float(table.s_values[i]), p=float(table.p_values[j]), rmsd=rmsd, log_likelihood=float(ll[i, j]),
        predicted=dict(zip(NAMING_CATEGORIES, pred.round(4).tolist())),
        posterior_mean=(s_mean, p_mean),
        s_interval=_interval(table.s_values, ps), p_interval=_interval(table.p_values, pp),
    )


def _interval(values: np.ndarray, marginal: np.ndarray, mass: float = 0.95) -> tuple[float, float]:
    cdf = np.cumsum(marginal)
    lo = values[int(np.searchsorted(cdf, (1 - mass) / 2))]
    hi = values[min(int(np.searchsorted(cdf, 1 - (1 - mass) / 2)), len(values) - 1)]
    return float(lo), float(hi)
