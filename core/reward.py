"""Proxy reward, decomposed into named terms (configs/reward.yaml).

Every term uses only what a real session produces (outcomes, tiers, cues), so the same reward could
be computed live. The true objective (learning gain) is hidden and only used for evaluation.

Per-trial terms are computed from the trace so far; the tier-climb bonus is paid once at the end
of the session and only if the session stayed in band. Components are returned separately so they
can be logged and ablated.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from core.config import ClinicalConfig, load_yaml

_STRICT = ConfigDict(extra="forbid")
TERMS = ("band", "climb", "frustration", "cue_waste")


class RewardTerms(BaseModel):
    model_config = _STRICT
    band: bool
    climb: bool
    frustration: bool
    cue_waste: bool


class RewardConfig(BaseModel):
    model_config = _STRICT
    variant: str
    variants: dict[str, RewardTerms]
    band_bonus: float = Field(ge=0.0)
    climb_scale: float = Field(ge=0.0)
    climb_min_band: float = Field(ge=0.0, le=1.0)
    frustration_penalty: float = Field(ge=0.0)
    cue_waste_penalty: float = Field(ge=0.0)
    heavy_cue_strength: int = Field(ge=1)

    @property
    def terms(self) -> RewardTerms:
        return self.variants[self.variant]


def load_reward(path: str | Path = "configs/reward.yaml", variant: str | None = None) -> RewardConfig:
    cfg = RewardConfig(**load_yaml(path))
    if variant is not None:
        cfg = cfg.model_copy(update={"variant": variant})
    if cfg.variant not in cfg.variants:
        raise ValueError(f"unknown reward variant {cfg.variant!r}; expected one of {list(cfg.variants)}")
    return cfg


class ProxyReward:
    def __init__(self, cfg: RewardConfig, clinical: ClinicalConfig) -> None:
        self.cfg, self.clinical = cfg, clinical

    def _in_band(self, trace: list[dict], upto: int) -> bool | None:
        """Whether rolling accuracy ending at trial `upto` is in band; None before a full window."""
        w = self.clinical.window
        if upto + 1 < w:
            return None
        acc = sum(t["outcome"] for t in trace[upto + 1 - w: upto + 1]) / w
        lo, hi = self.clinical.band
        return lo <= acc <= hi

    def step(self, trace: list[dict]) -> dict[str, float]:
        """Reward components for the latest trial in `trace` (trace[-1])."""
        t, terms, cfg = len(trace) - 1, self.cfg.terms, self.cfg
        out = dict.fromkeys(TERMS, 0.0)
        if terms.band and self._in_band(trace, t):
            out["band"] = cfg.band_bonus
        if terms.frustration:
            streak = 0
            for tr in reversed(trace):
                if tr["outcome"]:
                    break
                streak += 1
            if streak >= 3:
                out["frustration"] = -cfg.frustration_penalty
        if terms.cue_waste and t >= 1:
            prev, cur = trace[t - 1], trace[t]
            if (cur["outcome"] and cur["cue"] == "none" and prev["outcome"]
                    and self.clinical.cue_strength[prev["cue"]] >= cfg.heavy_cue_strength):
                out["cue_waste"] = -cfg.cue_waste_penalty
        return out

    def time_in_band(self, trace: list[dict]) -> float:
        flags = [self._in_band(trace, i) for i in range(len(trace))]
        flags = [f for f in flags if f is not None]
        return float(np.mean(flags)) if flags else 0.0

    def terminal(self, trace: list[dict], start_tier: int) -> dict[str, float]:
        """End-of-session components (tier climb, conditioned on staying in band)."""
        out = dict.fromkeys(TERMS, 0.0)
        if self.cfg.terms.climb and trace and self.time_in_band(trace) >= self.cfg.climb_min_band:
            out["climb"] = self.cfg.climb_scale * (trace[-1]["tier"] - start_tier)
        return out

    def episode(self, trace: list[dict], start_tier: int) -> dict[str, float]:
        """Total of every component over a whole session (for tests and evaluation)."""
        total = dict.fromkeys(TERMS, 0.0)
        for i in range(len(trace)):
            for k, v in self.step(trace[: i + 1]).items():
                total[k] += v
        for k, v in self.terminal(trace, start_tier).items():
            total[k] += v
        total["total"] = sum(total[k] for k in TERMS)
        return total
