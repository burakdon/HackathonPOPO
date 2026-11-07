"""Typed loading of the YAML configs.

All tunable numbers live in configs/*.yaml. These pydantic models validate them at load time,
so a typo in a config fails loudly at startup instead of silently changing behaviour.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from core.schema import CUES, TIERS, Cue

REPO_ROOT = Path(__file__).resolve().parents[1]

_STRICT = ConfigDict(extra="forbid")


class ClinicalConfig(BaseModel):
    model_config = _STRICT
    band: tuple[float, float]
    target: float
    window: int = Field(ge=1)
    max_tier_step: int = Field(ge=1, le=2)
    approved_cues: list[Cue]
    cue_strength: dict[Cue, int]

    @model_validator(mode="after")
    def _check(self) -> "ClinicalConfig":
        lo, hi = self.band
        if not 0.0 <= lo < self.target < hi <= 1.0:
            raise ValueError(f"need 0 <= band_lo < target < band_hi <= 1, got {self.band}, {self.target}")
        if "none" not in self.approved_cues:
            raise ValueError("approved_cues must include 'none' so the action mask is never empty")
        if set(self.cue_strength) != set(CUES):
            raise ValueError(f"cue_strength must define every cue {CUES}")
        return self


class BetaBanditConfig(BaseModel):
    model_config = _STRICT
    seed: int
    prior: dict[int, tuple[float, float]]
    tier_bonus: float = Field(ge=0.0)
    cue_cost: float = Field(ge=0.0)
    tolerance: float = Field(ge=0.0, le=0.5)
    discount: float = Field(gt=0.0, le=1.0)
    cue_only_after_error: bool

    @model_validator(mode="after")
    def _check(self) -> "BetaBanditConfig":
        if set(self.prior) != set(TIERS):
            raise ValueError(f"prior must define every tier {TIERS}")
        if any(a <= 0 or b <= 0 for a, b in self.prior.values()):
            raise ValueError("Beta prior parameters must be positive")
        return self


class LogisticBanditConfig(BaseModel):
    model_config = _STRICT
    seed: int
    # If set, the prior (mean vector + full covariance) is loaded from this generated file
    # (train/fit_bandit_prior.py) and prior_mean / prior_var are only the fallback.
    prior_file: str | None = None
    prior_mean: dict[str, float]
    prior_var: dict[Literal["action", "context", "interaction"], float]
    process_noise: float = Field(ge=0.0)
    tolerance: float = Field(ge=0.0, le=0.5)
    cue_cost: float = Field(ge=0.0)
    tier_bonus: float = Field(ge=0.0)
    cue_only_after_error: bool
    nominal_trials: int = Field(ge=1)
    rt_center_s: float
    rt_scale_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _check(self) -> "LogisticBanditConfig":
        if set(self.prior_var) != {"action", "context", "interaction"} or min(self.prior_var.values()) <= 0:
            raise ValueError("prior_var needs positive action, context and interaction entries")
        from agents.logistic_bandit import FEATURES  # local import: avoids a core -> agents cycle at import time
        unknown = set(self.prior_mean) - set(FEATURES)
        if unknown:
            raise ValueError(f"prior_mean has unknown features {sorted(unknown)}")
        return self


class LiveDittoConfig(BaseModel):
    model_config = _STRICT
    rounds: int = Field(ge=1)
    gradient_steps: int = Field(ge=1)
    learning_rate: float = Field(gt=0.0)
    beta: float = Field(gt=0.0)
    samples_per_state: int = Field(ge=1)
    anchor_weight: float = Field(ge=0.0)
    anchor_pool_patients: int = Field(ge=1)
    seed: int


class ServiceConfig(BaseModel):
    model_config = _STRICT
    agent: Literal["ppo", "logistic_bandit", "beta_bandit", "llm"]
    clinical_config: str
    bandit_config: str
    logistic_bandit_config: str
    ppo_model: str
    live_ditto: LiveDittoConfig
    default_session_id: str = Field(min_length=1)
    overrides_path: str
    cors_origins: list[str] = Field(default_factory=list)
    cors_origin_regex: str | None = None
    # Practice flow (service/practice.py) and the language-model policy (agents/llm_policy.py).
    practice_config: str = "configs/practice.yaml"
    # Model version /act uses when agent == "llm" (e.g. "ditto:<run_id>", "base", "prompted").
    llm_model: str | None = None


def resolve(path: str | Path) -> Path:
    """Resolve a config-relative path against the repo root (absolute paths pass through)."""
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def load_yaml(path: str | Path) -> dict:
    with open(resolve(path)) as f:
        return yaml.safe_load(f)


def load_clinical(path: str | Path = "configs/clinical.yaml") -> ClinicalConfig:
    return ClinicalConfig(**load_yaml(path))


def load_beta_bandit(path: str | Path = "configs/bandit.yaml") -> BetaBanditConfig:
    return BetaBanditConfig(**load_yaml(path))


def load_logistic_bandit(path: str | Path = "configs/logistic_bandit.yaml") -> LogisticBanditConfig:
    return LogisticBanditConfig(**load_yaml(path))


def load_service(path: str | Path = "configs/service.yaml") -> ServiceConfig:
    return ServiceConfig(**load_yaml(path))
