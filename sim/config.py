"""Typed view of configs/sim.yaml (validated at load so a typo fails loudly)."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.config import load_yaml
from core.schema import CUES, Cue

_STRICT = ConfigDict(extra="forbid")


class ItemsConfig(BaseModel):
    model_config = _STRICT
    path: str
    beta_frequency: float = Field(ge=0.0)
    beta_length: float = Field(ge=0.0)
    # If set and the CSV has an `irt_difficulty_t` column (T-score scale, as published for the
    # Philadelphia Naming Test), item difficulty is the MEASURED one: irt_slope * (b - mean b),
    # in log-odds, instead of the frequency/length formula.
    irt_slope: float | None = Field(default=None, gt=0.0)


class CuesConfig(BaseModel):
    model_config = _STRICT
    phonemic_weight: dict[Cue, float]
    semantic_weight: dict[Cue, float]

    @model_validator(mode="after")
    def _all_cues(self) -> "CuesConfig":
        for name, w in (("phonemic_weight", self.phonemic_weight), ("semantic_weight", self.semantic_weight)):
            if set(w) != set(CUES):
                raise ValueError(f"{name} must define every cue {CUES}")
        if self.phonemic_weight["none"] != 0 or self.semantic_weight["none"] != 0:
            raise ValueError("cue 'none' must have zero gain")
        return self


class FatigueConfig(BaseModel):
    model_config = _STRICT
    streak_gain: float = Field(ge=0.0)
    cap: float = Field(gt=0.0)


class LearningConfig(BaseModel):
    model_config = _STRICT
    peak: float = Field(gt=0.0, lt=1.0)
    sharpness: float = Field(gt=1.0)
    cue_discount: float = Field(ge=0.0, le=1.0)
    generalisation: float = Field(ge=0.0, le=1.0)

    @property
    def a(self) -> float:
        return self.peak * self.sharpness

    @property
    def b(self) -> float:
        return (1.0 - self.peak) * self.sharpness


class ResponseTimeConfig(BaseModel):
    model_config = _STRICT
    base_s: float = Field(gt=0.0)
    difficulty_gain: float
    fatigue_gain: float
    sigma: float = Field(ge=0.0)
    fail_extra_s: float = Field(ge=0.0)
    overhead_s: float = Field(ge=0.0)


class DropoutConfig(BaseModel):
    model_config = _STRICT
    enabled: bool
    base: float = Field(ge=0.0, le=1.0)
    streak_gain: float = Field(ge=0.0)


class ProbeConfig(BaseModel):
    model_config = _STRICT
    enabled: bool
    every: int = Field(ge=2)
    min_gap: int = Field(ge=1)


class SessionConfig(BaseModel):
    model_config = _STRICT
    n_trials: int = Field(ge=1)
    start_tier: int = Field(ge=1, le=3)


class ParamDist(BaseModel):
    """Normal(mean, sd) truncated to [low, high] (sampled by rejection; see sim/population.py)."""

    model_config = _STRICT
    mean: float
    sd: float = Field(ge=0.0)
    low: float
    high: float

    @model_validator(mode="after")
    def _bounds(self) -> "ParamDist":
        if not self.low <= self.mean <= self.high:
            raise ValueError(f"need low <= mean <= high, got {self}")
        return self


class TwoStepConfig(BaseModel):
    """Two-step patient model (sim/two_step.py). Cue weights are per step, before the patient's
    responsiveness multipliers; frequency acts on step 1 (word selection), length on step 2 (sounds)."""

    model_config = _STRICT
    beta_frequency: float = Field(ge=0.0)
    beta_length: float = Field(ge=0.0)
    stage1_cue: dict[Cue, float]
    stage2_cue: dict[Cue, float]
    # Training population: `parametric` (the `population` distributions) or `pnt`: each patient's
    # overall naming ability is drawn from real patients' Philadelphia Naming Test ability scores
    # (`thetas_path`, T-scale), and only the split of that difficulty between the two steps is
    # sampled (profile_beta: Beta(a, b) share of the failure on step 1).
    population_source: Literal["parametric", "pnt"] = "parametric"
    thetas_path: str | None = None
    theta_jitter: float = Field(default=2.0, ge=0.0)    # T-points of noise around each real score
    profile_beta: tuple[float, float] = (2.0, 2.0)

    @model_validator(mode="after")
    def _all_cues(self) -> "TwoStepConfig":
        if self.population_source == "pnt" and not self.thetas_path:
            raise ValueError("population_source: pnt needs thetas_path")
        for name, w in (("stage1_cue", self.stage1_cue), ("stage2_cue", self.stage2_cue)):
            if set(w) != set(CUES) or w["none"] != 0:
                raise ValueError(f"{name} must define every cue {CUES}, with 'none' = 0")
        return self


class SimConfig(BaseModel):
    model_config = _STRICT
    # Patient model: `logistic` (sim/patient.py, one log-odds score per trial) or `two_step`
    # (sim/two_step.py, meaning -> word -> sounds; errors and cue effects follow from where retrieval fails).
    model: Literal["logistic", "two_step"] = "logistic"
    two_step: TwoStepConfig | None = None
    items: ItemsConfig
    cues: CuesConfig
    fatigue: FatigueConfig
    learning: LearningConfig
    errors: dict = Field(default_factory=dict)
    response_time: ResponseTimeConfig
    dropout: DropoutConfig
    review_probes: ProbeConfig
    session: SessionConfig
    parameters: list[str]
    population: dict[str, ParamDist]
    presets: dict[str, dict[str, ParamDist]]
    splits: dict[str, int]

    @model_validator(mode="after")
    def _complete(self) -> "SimConfig":
        needed = set(self.parameters)
        for name, dists in [("population", self.population), *self.presets.items()]:
            if set(dists) != needed:
                raise ValueError(f"{name} must define exactly {sorted(needed)}")
        return self


def load_sim(path: str | Path | None = None) -> SimConfig:
    """Default: configs/sim.yaml, or the file in the SIM_CONFIG environment variable, so a whole
    pipeline (training, evaluation, service) can switch patient model without code changes:
        SIM_CONFIG=configs/sim_two_step.yaml uv run python -m train.train_ppo"""
    cfg = SimConfig(**load_yaml(path or os.environ.get("SIM_CONFIG", "configs/sim.yaml")))
    if cfg.model == "two_step" and cfg.two_step is None:
        raise ValueError("model: two_step needs a `two_step:` section")
    return cfg
