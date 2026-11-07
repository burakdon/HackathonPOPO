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
from sim.config import ItemsConfig

REQUIRED_COLUMNS = ("word", "image_file", "frequency", "length", "tier")


@dataclass(frozen=True)
class ItemBank:
    words: tuple[str, ...]
    image_files: tuple[str, ...]
    frequency: np.ndarray
    length: np.ndarray
    tier: np.ndarray
    difficulty: np.ndarray  # log-odds units, higher = harder

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
    return ItemBank(words, tuple(r["image_file"] for r in rows), frequency, length, tier, difficulty)


class ItemSelector:
    """Unseen-first, then least-recently-seen item within the requested tier."""

    def __init__(self, bank: ItemBank, rng: np.random.Generator) -> None:
        self.bank, self.rng = bank, rng
        self.last_seen = np.full(len(bank), -1)  # trial index of last presentation, -1 = never
        self.t = 0

    def next(self, tier: int) -> int:
        idx = self.bank.indices_in_tier(tier)
        seen = self.last_seen[idx]
        candidates = idx[seen == seen.min()]
        i = int(self.rng.choice(candidates))
        self.last_seen[i] = self.t
        self.t += 1
        return i
