"""Typed view of configs/sim.yaml (validated at load so a typo fails loudly)."""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.config import load_yaml
from core.schema import CUES, Cue

_STRICT = ConfigDict(extra="forbid")


class ItemsConfig(BaseModel):
    model_config = _STRICT
    path: str
    beta_frequency: float = Field(ge=0.0)
    beta_length: float = Field(ge=0.0)


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


class SimConfig(BaseModel):
    model_config = _STRICT
    items: ItemsConfig
    cues: CuesConfig
    fatigue: FatigueConfig
    learning: LearningConfig
    errors: dict = Field(default_factory=dict)
    response_time: ResponseTimeConfig
    dropout: DropoutConfig
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


def load_sim(path: str | Path = "configs/sim.yaml") -> SimConfig:
    return SimConfig(**load_yaml(path))
