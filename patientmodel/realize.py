"""Turn an SP-model response category into an actual response for a given target word.

The SP model says *what kind* of error happens. This module says *which word* (or nonword) comes
out, using the lexicon's semantic similarity, the PNT formal-relatedness criterion, and phoneme
edits for nonwords. Candidate words are weighted by relatedness and by frequency, as real errors
are. Nonword edits default to the distribution measured on PSST nonword-like responses
(see `NonwordEdits`); a mixed error falls back to a semantic error when the target has no mixed
neighbour in the lexicon, mirroring the model's own 20% mixed-opportunity assumption.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from patientmodel.lexicon import VOWELS, Lexicon, formally_related, phoneme_edit_distance

CONSONANTS = ["B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH"]


@dataclass(frozen=True)
class RealizerConfig:
    semantic_top_k: int = 15
    semantic_min_similarity: float = 0.35
    formal_top_k: int = 15
    unrelated_pool: int = 2000           # draw unrelated words from the most frequent N
    frequency_weight: float = 0.5        # candidate weight ~ relatedness * exp(freq_weight * zipf)
    # Per-phoneme edit rate for nonwords (binomial in word length, at least one edit). PSST nonword-
    # like responses to 5-9 phoneme targets sit 1-6 edits away, mode about 4; 0.4 reproduces that
    # spread for long words and gives mostly 1-2 edits for CVC words. To be refined against PSST.
    nonword_edit_rate: float = 0.4


@dataclass
class Realization:
    category: str
    word: str | None                     # None for omissions
    phonemes: tuple[str, ...] | None


@dataclass
class Realizer:
    lexicon: Lexicon
    cfg: RealizerConfig = field(default_factory=RealizerConfig)

    # ---------------------------------------------------------------- candidate sets
    def _morph_variant(self, a: str, b: str) -> bool:
        return a == b or a.startswith(b) or b.startswith(a) or (len(a) > 4 and len(b) > 4 and a[:4] == b[:4])

    def semantic_candidates(self, target: str) -> tuple[list[int], np.ndarray]:
        lex = self.lexicon
        sim = lex.similarity(target)
        order = np.argsort(-sim)
        tp = lex.pron(target)
        out, w = [], []
        for i in order[1: self.cfg.semantic_top_k * 4]:
            cand = lex.words[i]
            if sim[i] < self.cfg.semantic_min_similarity or self._morph_variant(cand, target):
                continue
            if formally_related(tp, lex.prons[i]):
                continue                      # that is a mixed error, not a pure semantic one
            out.append(int(i)); w.append(sim[i])
            if len(out) >= self.cfg.semantic_top_k:
                break
        return out, np.array(w)

    def formal_candidates(self, target: str) -> tuple[list[int], np.ndarray]:
        lex = self.lexicon
        tp = lex.pron(target)
        sim = lex.similarity(target)
        scored = []
        for i, p in enumerate(lex.prons):
            if lex.words[i] == target or self._morph_variant(lex.words[i], target):
                continue
            if abs(len(p) - len(tp)) > 2 or not formally_related(tp, p):
                continue
            if sim[i] >= self.cfg.semantic_min_similarity:
                continue                      # mixed, not pure formal
            scored.append((phoneme_edit_distance(tp, p), i))
        scored.sort()
        idx = [i for _, i in scored[: self.cfg.formal_top_k]]
        w = np.array([1.0 / (1 + d) for d, _ in scored[: self.cfg.formal_top_k]])
        return idx, w

    def mixed_candidates(self, target: str) -> tuple[list[int], np.ndarray]:
        lex = self.lexicon
        tp = lex.pron(target)
        sim = lex.similarity(target)
        out, w = [], []
        for i in np.argsort(-sim)[1:400]:
            if sim[i] < self.cfg.semantic_min_similarity:
                break
            cp = lex.prons[i]
            if self._morph_variant(lex.words[i], target) or not formally_related(tp, cp):
                continue
            # A mixed error shares real phonology, not just one stray phoneme: same onset, or close.
            if cp[0] != tp[0] and phoneme_edit_distance(tp, cp) > 2:
                continue
            out.append(int(i)); w.append(sim[i])
        return out, np.array(w)

    def _pick(self, idx: list[int], w: np.ndarray, rng: np.random.Generator) -> int:
        weights = w * np.exp(self.cfg.frequency_weight * self.lexicon.log_freq[idx])
        return idx[rng.choice(len(idx), p=weights / weights.sum())]

    # ---------------------------------------------------------------- nonwords
    def nonword(self, target: str, rng: np.random.Generator) -> tuple[str, ...]:
        """Phoneme edits of the target. The number of edits scales with length (binomial with
        per-phoneme rate `nonword_edit_rate`, at least one); substitutions stay within class
        (vowel for vowel, consonant for consonant); the result must not be a word, must keep at
        least one vowel, and must not create identical adjacent phonemes or a 3-consonant onset."""
        lex = self.lexicon
        tp = list(lex.pron(target))
        for _ in range(40):
            n_edits = max(1, int(rng.binomial(len(tp), self.cfg.nonword_edit_rate)))
            out = list(tp)
            for _ in range(n_edits):
                op = rng.choice(["sub", "sub", "sub", "del", "ins"]) if len(out) > 3 else "sub"
                k = int(rng.integers(len(out)))
                if op == "sub":
                    pool = sorted(VOWELS) if out[k] in VOWELS else CONSONANTS
                    out[k] = str(rng.choice([x for x in pool if x != out[k]]))
                elif op == "del":
                    del out[k]
                else:
                    out.insert(k, str(rng.choice(CONSONANTS)))
            res = tuple(out)
            if res == tuple(tp) or lex.is_word(res) or not any(x in VOWELS for x in res):
                continue
            if any(a == b for a, b in zip(res, res[1:])):
                continue
            onset = 0
            for x in res:
                if x in VOWELS:
                    break
                onset += 1
            if onset > 2:
                continue
            return res
        out = list(tp); out[0] = "B" if out[0] != "B" else "D"
        return tuple(out)

    # ---------------------------------------------------------------- main entry
    def realize(self, target: str, category: str, rng: np.random.Generator) -> Realization:
        lex = self.lexicon
        if lex.pron(target) is None:
            raise KeyError(f"{target!r} is not in the lexicon")
        if category == "correct":
            return Realization("correct", target, lex.pron(target))
        if category == "omission":
            return Realization("omission", None, None)
        if category == "nonword":
            ph = self.nonword(target, rng)
            return Realization("nonword", "".join(p.lower() for p in ph), ph)
        if category == "mixed":
            idx, w = self.mixed_candidates(target)
            if not idx:
                category = "semantic"
        if category == "semantic":
            idx, w = self.semantic_candidates(target)
            if not idx:
                category = "unrelated"
        if category == "formal":
            idx, w = self.formal_candidates(target)
            if not idx:
                return self.realize(target, "nonword", rng)
        if category == "unrelated":
            tp = lex.pron(target); sim = lex.similarity(target)
            pool = [i for i in np.argsort(-lex.log_freq)[: self.cfg.unrelated_pool]
                    if sim[i] < self.cfg.semantic_min_similarity and not formally_related(tp, lex.prons[i])
                    and not self._morph_variant(lex.words[i], target)]
            idx, w = pool, np.ones(len(pool))
        i = self._pick(idx, w, rng)
        return Realization(category, lex.words[i], lex.prons[i])
