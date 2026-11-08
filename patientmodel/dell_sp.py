"""The semantic-phonological (SP) interactive two-step model of lexical access in naming.

Foygel & Dell (2000). Architecture and constants: Dell, Lawler, Harris & Gordon (2004, pp. 127-128)
and Walker (2016, dissertation, pp. 37-38 and 101), which spells out the implemented lexicon:

  six words: the target (CAT), one semantic neighbour sharing 3 of the target's 10 semantic
  features (DOG), two formal neighbours that differ from the target only in their onset (MAT, HAT),
  and two unrelated words sharing nothing with the target (LOG, FOG). Ten phonological units:
  6 onsets, 2 vowels, 2 codas. On a fraction of trials (20%) one formal neighbour also shares 3
  semantic features with the target, making it a mixed neighbour (RAT).

A patient is two numbers: s, the weight of semantic<->word connections, and p, the weight of
word<->phoneme connections. Normal is s = p = 0.04. Lower s yields word errors (semantic, mixed,
unrelated), lower p yields nonwords.

Naming one picture:
  1. boost the target's semantic features (10 units each); spread for 8 steps; pick the most
     active word
  2. boost that word (100 units); spread for 8 more steps; pick the most active onset, vowel, coda
  3. classify the output string: target -> correct, DOG -> semantic, MAT/HAT -> formal, RAT ->
     mixed, LOG/FOG -> unrelated, any other string -> nonword
Activation rule (Dell et al. 2004): a[j,t] = a[j,t-1](1-decay) + sum_i w[i,j] max(a[i,t-1], 0) + noise,
noise = N(0, .01) + N(0, .16 * a[j,t]) applied after the update; negative activation does not spread
(Walker 2016, p. 101). With these constants the model reproduces the SP-model proportions Dell
published for fitted patients (Dell et al. 2004, Table 2; 2013, Table 1) to within 0.01.
All trials are simulated in parallel with numpy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CATEGORIES: tuple[str, ...] = ("correct", "semantic", "formal", "mixed", "unrelated", "nonword", "omission")
NAMING_CATEGORIES: tuple[str, ...] = CATEGORIES[:6]   # the six PNT categories the model is fitted on

# Roles in the six-word neighbourhood. "formal_b" becomes the mixed neighbour on mixed trials.
WORDS: tuple[str, ...] = ("target", "semantic", "formal_a", "formal_b", "unrelated_a", "unrelated_b")
# Phonology: (onset index, vowel index, coda index). Formal neighbours share the target's vowel and
# coda and differ in onset; the other three words use the other vowel and coda.
PHONOLOGY: dict[str, tuple[int, int, int]] = {
    "target": (0, 0, 0), "formal_a": (1, 0, 0), "formal_b": (2, 0, 0),
    "semantic": (3, 1, 1), "unrelated_a": (4, 1, 1), "unrelated_b": (5, 1, 1),
}
N_ONSETS, N_VOWELS, N_CODAS = 6, 2, 2


@dataclass(frozen=True)
class DellSPConfig:
    decay: float = 0.6
    intrinsic_noise_sd: float = 0.01
    activation_noise_sd: float = 0.16
    steps_per_stage: int = 8
    semantic_jolt: float = 10.0
    lexical_jolt: float = 100.0
    normal_weight: float = 0.04
    weight_max: float = 0.06
    features_per_word: int = 10
    shared_features: int = 3
    mixed_opportunity: float = 0.2
    omission_threshold: float | None = None

    def __post_init__(self) -> None:
        if not 0.0 < self.decay < 1.0:
            raise ValueError("decay must be in (0, 1)")
        if 2 * self.shared_features > self.features_per_word:
            raise ValueError("the semantic and mixed neighbours cannot share more features than the target has")
        if not 0.0 <= self.mixed_opportunity <= 1.0:
            raise ValueError("mixed_opportunity is a probability")


def load_dell_sp_config(path: str | Path = "configs/dell_sp.yaml") -> tuple[DellSPConfig, dict]:
    """Return the model config and the raw `fit` section."""
    with open(path) as f:
        raw = yaml.safe_load(f)
    fit = raw.pop("fit", {})
    return DellSPConfig(**raw), fit


class _Network:
    """Unit layout and connection masks for one neighbourhood (with or without the mixed word)."""

    def __init__(self, cfg: DellSPConfig, mixed: bool) -> None:
        f = cfg.features_per_word
        feats: dict[str, list[int]] = {}
        n = 0
        target = list(range(n, n + f)); n += f
        feats["target"] = target
        # Semantic neighbour shares the first `shared` target features; the mixed neighbour (if any)
        # shares the next `shared`, so the two neighbours do not share features with each other.
        own = f - cfg.shared_features
        feats["semantic"] = target[: cfg.shared_features] + list(range(n, n + own)); n += own
        for w in ("formal_a", "formal_b", "unrelated_a", "unrelated_b"):
            if mixed and w == "formal_b":
                feats[w] = target[cfg.shared_features: 2 * cfg.shared_features] + list(range(n, n + own)); n += own
            else:
                feats[w] = list(range(n, n + f)); n += f
        self.n_features = n
        self.word_start = n
        self.word_index = {w: n + i for i, w in enumerate(WORDS)}
        n += len(WORDS)
        self.onset_ids = np.arange(n, n + N_ONSETS); n += N_ONSETS
        self.vowel_ids = np.arange(n, n + N_VOWELS); n += N_VOWELS
        self.coda_ids = np.arange(n, n + N_CODAS); n += N_CODAS
        self.n_units = n
        self.target_features = np.array(feats["target"])

        self.sem_mask = np.zeros((n, n)); self.phon_mask = np.zeros((n, n))
        for w in WORDS:
            wi = self.word_index[w]
            for fi in feats[w]:
                self.sem_mask[fi, wi] = self.sem_mask[wi, fi] = 1.0
            o, v, c = PHONOLOGY[w]
            for pi in (self.onset_ids[o], self.vowel_ids[v], self.coda_ids[c]):
                self.phon_mask[wi, pi] = self.phon_mask[pi, wi] = 1.0

        # Output string -> category over all onset x vowel x coda combinations.
        cat_of = {"target": "correct", "semantic": "semantic", "formal_a": "formal",
                  "formal_b": "mixed" if mixed else "formal", "unrelated_a": "unrelated", "unrelated_b": "unrelated"}
        self.string_category = np.full((N_ONSETS, N_VOWELS, N_CODAS), NAMING_CATEGORIES.index("nonword"))
        for w, (o, v, c) in PHONOLOGY.items():
            self.string_category[o, v, c] = NAMING_CATEGORIES.index(cat_of[w])

    def weights(self, s: float, p: float) -> np.ndarray:
        return s * self.sem_mask + p * self.phon_mask


@dataclass(frozen=True)
class CueInput:
    """External activation a cue adds to the target's units (activation units, same scale as the
    jolts). Semantic and word input are applied once at the start of word access; phonological input
    (onset, vowel, coda of the target) is applied at the start of both stages, since a spoken cue is
    still available while the word's sounds are being assembled. Because the formal neighbours share
    the target's vowel and coda but not its onset (PHONOLOGY), an onset cue singles out the target
    while a coda cue also helps the formal competitors."""
    semantic: float = 0.0
    onset: float = 0.0
    vowel: float = 0.0
    coda: float = 0.0
    word: float = 0.0


class DellSPModel:
    """Vectorised simulator over the two neighbourhoods. Immutable after construction."""

    def __init__(self, cfg: DellSPConfig | None = None) -> None:
        self.cfg = cfg or DellSPConfig()
        self.plain = _Network(self.cfg, mixed=False)
        self.mixed = _Network(self.cfg, mixed=True)

    def _spread(self, a: np.ndarray, W: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        c = self.cfg
        for _ in range(c.steps_per_stage):
            pos = np.maximum(a, 0.0)                     # negative activation does not spread
            new = a * (1.0 - c.decay) + pos @ W
            # Noise is applied after the decay-and-input update (Walker 2016, p. 101), and the
            # activation-dependent term scales with the updated activation a[j,t] (Dell et al. 2004).
            # Scaling it with the previous activation instead roughly doubles the effective noise and
            # breaks the published fits (see tests/test_dell_sp.py::test_reproduces_published_fits).
            new += rng.normal(0.0, c.intrinsic_noise_sd, a.shape)
            new += rng.normal(0.0, 1.0, a.shape) * c.activation_noise_sd * np.maximum(new, 0.0)
            a = new
        return a

    def _phon_cue(self, net: _Network, a: np.ndarray, cue: CueInput) -> None:
        o, v, cd = PHONOLOGY["target"]
        a[:, net.onset_ids[o]] += cue.onset
        a[:, net.vowel_ids[v]] += cue.vowel
        a[:, net.coda_ids[cd]] += cue.coda

    def _run(self, net: _Network, s: float, p: float, n: int, rng: np.random.Generator,
             cue: CueInput | None = None) -> np.ndarray:
        c = self.cfg
        if n == 0:
            return np.zeros(0, dtype=int)
        W = net.weights(s, p)
        a = np.zeros((n, net.n_units))
        a[:, net.target_features] += c.semantic_jolt
        if cue is not None:
            a[:, net.target_features] += cue.semantic
            a[:, net.word_index["target"]] += cue.word
            self._phon_cue(net, a, cue)
        a = self._spread(a, W, rng)
        word_act = a[:, net.word_start: net.word_start + len(WORDS)]
        chosen = word_act.argmax(axis=1)
        omitted = (word_act.max(axis=1) < c.omission_threshold) if c.omission_threshold is not None \
            else np.zeros(n, dtype=bool)
        a[np.arange(n), net.word_start + chosen] += c.lexical_jolt
        if cue is not None:
            self._phon_cue(net, a, cue)
        a = self._spread(a, W, rng)
        o = a[:, net.onset_ids].argmax(axis=1)
        v = a[:, net.vowel_ids].argmax(axis=1)
        cd = a[:, net.coda_ids].argmax(axis=1)
        cat = net.string_category[o, v, cd].copy()
        cat[omitted] = CATEGORIES.index("omission")
        return cat

    def simulate_trials(self, s: float, p: float, n_trials: int, rng: np.random.Generator,
                        cue: CueInput | None = None) -> np.ndarray:
        """Int array of category indices (into CATEGORIES) for n_trials namings. Mixed-neighbourhood
        trials are drawn with probability `mixed_opportunity`; the output order is shuffled."""
        n_mixed = int(rng.binomial(n_trials, self.cfg.mixed_opportunity))
        cat = np.concatenate([self._run(self.plain, s, p, n_trials - n_mixed, rng, cue),
                              self._run(self.mixed, s, p, n_mixed, rng, cue)])
        rng.shuffle(cat)
        return cat

    def simulate(self, s: float, p: float, n_trials: int = 4000, seed: int | None = 0,
                 cue: CueInput | None = None) -> dict[str, float]:
        """Category probabilities for a patient with weights (s, p), optionally under a cue."""
        rng = np.random.default_rng(seed)
        counts = np.bincount(self.simulate_trials(s, p, n_trials, rng, cue), minlength=len(CATEGORIES))
        return {k: float(v) / n_trials for k, v in zip(CATEGORIES, counts)}

    def name_once(self, s: float, p: float, rng: np.random.Generator) -> str:
        """One naming attempt; returns the category name. For use inside a session simulator."""
        return CATEGORIES[int(self.simulate_trials(s, p, 1, rng)[0])]
