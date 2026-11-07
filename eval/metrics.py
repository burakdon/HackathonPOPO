"""Per-session evaluation metrics.

Observable metrics (computable on a real patient): time in band, accuracy, tier progression,
frustration events, cue efficiency. Simulation-only metrics use hidden state: learning gain (the
true objective) and control error |p_true - target|.
"""
from __future__ import annotations

import numpy as np

from core.config import ClinicalConfig
from sim.session import SessionResult


def rolling_accuracy(outcomes: list[bool], window: int) -> np.ndarray:
    """Rolling accuracy over the last `window` trials; only full windows (index window-1 onward)."""
    a = np.asarray(outcomes, dtype=float)
    if len(a) < window:
        return np.array([])
    c = np.cumsum(np.insert(a, 0, 0.0))
    return (c[window:] - c[:-window]) / window


def session_metrics(result: SessionResult, clinical: ClinicalConfig) -> dict[str, float]:
    trace = result.trace
    outcomes = [t["outcome"] for t in trace]
    roll = rolling_accuracy(outcomes, clinical.window)
    lo, hi = clinical.band
    streak = frustration = 0
    for o in outcomes:
        streak = 0 if o else streak + 1
        frustration += int(streak == 3)   # each run of >= 3 failures counts once
    tiers = np.array([t["tier"] for t in trace])
    correct_cue_strength = [clinical.cue_strength[t["cue"]] for t in trace if t["outcome"]]
    probes = [t for t in trace if t.get("probe")]
    return {
        "time_in_band": float(np.mean((roll >= lo) & (roll <= hi))) if len(roll) else 0.0,
        "accuracy": float(np.mean(outcomes)),
        "mean_tier": float(tiers.mean()),
        "final_tier": float(tiers[-1]),
        "tier_progression": float(tiers[-1] - result.start_tier),
        "frac_tier1": float(np.mean(tiers == 1)),
        "frustration_events": float(frustration),
        "cue_strength_on_correct": float(np.mean(correct_cue_strength)) if correct_cue_strength else 0.0,
        "learning_gain": float(result.learning_gain),
        "control_error": float(np.mean([abs(t["p_true"] - clinical.target) for t in trace])),
        "n_trials": float(len(trace)),
        "quit": float(result.quit),
        # Objective: later naming without hints. Unassisted recall on review probes (observable);
        # mean true p on probes (sim-only); probe lift = improvement of the probed items' uncued
        # recall since session start (sim-only; not inflated by practising only easy items).
        # NaN if no probe happened.
        "probe_recall": float(np.mean([t["outcome"] for t in probes])) if probes else float("nan"),
        "probe_p": float(np.mean([t["p_true"] for t in probes])) if probes else float("nan"),
        "probe_lift": float(np.mean([t["probe_lift"] for t in probes])) if probes else float("nan"),
        "n_probes": float(len(probes)),
    }


def mean_ci(values: list[float] | np.ndarray, z: float = 1.96) -> tuple[float, float]:
    """Mean and normal-approximation 95% CI half-width over non-NaN values."""
    v = np.asarray(values, dtype=float)
    v = v[~np.isnan(v)]
    if len(v) < 2:
        return float(v.mean()), float("nan")
    return float(v.mean()), float(z * v.std(ddof=1) / np.sqrt(len(v)))
