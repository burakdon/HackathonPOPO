"""Picture-naming item bank: loading, difficulty, and within-tier item selection.

CSV schema (one row per picture, header required):

    word        str    target name, unique
    image_file  str    path to the picture, relative to the app's image root
    frequency   float  word frequency on the Zipf scale (log10 per billion words), higher = more common
    length      int    word length; phonemes preferred, letters acceptable (z-scored within the bank,
                       so only consistency matters)
    tier        int    1..3, clinician-assigned difficulty tier

The agent chooses a tier; the item within the tier is chosen by `ItemSelector` (unseen this session
first, then least recently seen, ties broken at random). The frontend should use the same rule so
the simulator matches the live app.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass

import numpy as np

from core.config import resolve
from core.schema import TIERS
from sim.config import ItemsConfig, ProbeConfig

REQUIRED_COLUMNS = ("word", "image_file", "frequency", "length", "tier")


@dataclass(frozen=True)
class ItemBank:
    words: tuple[str, ...]
    image_files: tuple[str, ...]
    frequency: np.ndarray
    length: np.ndarray
    tier: np.ndarray
    difficulty: np.ndarray  # log-odds units, higher = harder
    irt_logit: np.ndarray | None = None   # measured difficulty (log-odds, centred), if the bank has it

    def __len__(self) -> int:
        return len(self.words)

    def indices_in_tier(self, tier: int) -> np.ndarray:
        return np.flatnonzero(self.tier == tier)


def _z(v: np.ndarray) -> np.ndarray:
    sd = v.std()
    return (v - v.mean()) / sd if sd > 0 else np.zeros_like(v)


def load_items(cfg: ItemsConfig) -> ItemBank:
    with open(resolve(cfg.path), newline="") as f:
        reader = csv.DictReader(f)
        missing = set(REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"item CSV {cfg.path} is missing columns {sorted(missing)}")
        rows = list(reader)
    words = tuple(r["word"].strip() for r in rows)
    if len(set(words)) != len(words):
        raise ValueError("item words must be unique")
    tier = np.array([int(r["tier"]) for r in rows])
    if not set(tier) <= set(TIERS) or any(not (tier == t).any() for t in TIERS):
        raise ValueError(f"every tier in {TIERS} needs at least one item, and no other tiers are allowed")
    frequency = np.array([float(r["frequency"]) for r in rows])
    length = np.array([float(r["length"]) for r in rows])
    difficulty = cfg.beta_frequency * -_z(frequency) + cfg.beta_length * _z(length)
    irt = None
    if cfg.irt_slope is not None:
        if "irt_difficulty_t" not in (reader.fieldnames or []):
            raise ValueError(f"items.irt_slope is set but {cfg.path} has no irt_difficulty_t column")
        b = np.array([float(r["irt_difficulty_t"]) for r in rows])
        irt = cfg.irt_slope * (b - b.mean())
        difficulty = irt
    return ItemBank(words, tuple(r["image_file"] for r in rows), frequency, length, tier, difficulty, irt)


class ItemSelector:
    """Chooses the item for every trial; the agent only chooses tier and cue.

    Practice trials: within the requested tier, an item not yet shown this session, else the least
    recently shown one (ties broken by the seeded rng).

    Review probes (objective: later naming without hints): trial k (0-based, counting all trials)
    is a probe if probes are enabled, (k + 1) is a multiple of `every`, and an eligible item exists.
    Eligible = practised earlier this session (not as a probe) and last shown at least `min_gap`
    trials ago. Among eligible items: fewest probes so far, then the one first practised earliest.
    This rule is deterministic so the frontend can reproduce it exactly (see README Interfaces).
    """

    def __init__(self, bank: ItemBank, rng: np.random.Generator, probes: "ProbeConfig | None" = None) -> None:
        self.bank, self.rng, self.probes = bank, rng, probes
        n = len(bank)
        self.last_seen = np.full(n, -1)          # trial index of last presentation (any kind), -1 = never
        self.first_practice = np.full(n, -1)     # trial index of first practice presentation
        self.probe_count = np.zeros(n, dtype=int)
        self.t = 0                               # index of the next trial

    def _eligible(self) -> np.ndarray:
        practised = self.first_practice >= 0
        return np.flatnonzero(practised & (self.t - self.last_seen >= self.probes.min_gap))

    def probe_due(self) -> bool:
        p = self.probes
        return bool(p is not None and p.enabled and (self.t + 1) % p.every == 0 and len(self._eligible()))

    def next_probe(self) -> int:
        cand = self._eligible()
        i = int(min(cand, key=lambda j: (self.probe_count[j], self.first_practice[j])))
        self.probe_count[i] += 1
        self.last_seen[i] = self.t
        self.t += 1
        return i

    def next(self, tier: int) -> int:
        idx = self.bank.indices_in_tier(tier)
        seen = self.last_seen[idx]
        candidates = idx[seen == seen.min()]
        i = int(self.rng.choice(candidates))
        if self.first_practice[i] < 0:
            self.first_practice[i] = self.t
        self.last_seen[i] = self.t
        self.t += 1
        return i
