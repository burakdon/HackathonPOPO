"""Turn an SP-model response category into an actual response for a given target word.

The SP model says *what kind* of error happens. This module says *which word* (or nonword) comes
out. Candidates are drawn from a curated noun lexicon and weighted by relatedness and frequency.

  semantic   a noun related in meaning: embedding cosine >= `semantic_min_similarity` AND a WordNet
             taxonomic relation (Wu-Palmer >= `wordnet_min_wup`), or a very high cosine; not formally
             related to the target
  formal     a noun meeting the PNT formal-relatedness criterion, weighted by phoneme edit distance;
             not semantically related
  mixed      both: semantic by the rule above AND sharing the onset or within two phoneme edits
  unrelated  a frequent noun sharing neither meaning nor phonology
  nonword    phoneme edits of the target that (a) are not any English word in CMUdict and (b) are
             phonotactically legal (lexicon.is_phonotactically_legal)
When a category has no candidate for a target (e.g. no mixed neighbour exists, as for 80% of real
trials) the realizer falls back: mixed -> semantic -> unrelated; formal -> nonword. The returned
category is the realized one; `fallback_from` records the model's original category.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from patientmodel.lexicon import (CONSONANTS, VOWELS, Lexicon, formally_related, is_phonotactically_legal,
                                  phoneme_edit_distance, wordnet)


@dataclass(frozen=True)
class RealizerConfig:
    semantic_top_k: int = 15
    semantic_min_similarity: float = 0.40
    semantic_high_similarity: float = 0.60   # above this, accept without a WordNet relation
    wordnet_min_wup: float = 0.50
    formal_top_k: int = 15
    unrelated_pool: int = 1500
    frequency_weight: float = 0.5
    # Per-phoneme substitution rate for nonwords (binomial in word length, at least one edit).
    # ASSUMPTION pending category-coded PSST data: real phonological paraphasias to short words are
    # mostly single-phoneme substitutions; to long words they are often several.
    nonword_edit_rate: float = 0.35
    nonword_indel_prob: float = 0.15          # share of edits that are deletions/insertions rather than substitutions


@dataclass
class Realization:
    category: str
    word: str | None                     # None for omissions
    phonemes: tuple[str, ...] | None
    fallback_from: str | None = None     # the model's category when the realizer had to substitute


@dataclass
class Realizer:
    lexicon: Lexicon
    cfg: RealizerConfig = field(default_factory=RealizerConfig)

    def __post_init__(self) -> None:
        self._wn = wordnet()
        self._sem_cache: dict[str, tuple[list[int], np.ndarray]] = {}
        self._form_cache: dict[str, tuple[list[int], np.ndarray]] = {}
        self._mixed_cache: dict[str, tuple[list[int], np.ndarray]] = {}
        self._unrel_cache: dict[str, list[int]] = {}

    # ---------------------------------------------------------------- helpers
    @staticmethod
    def _morph_variant(a: str, b: str) -> bool:
        """Same word in another form: plural, diminutive, compound containing the other."""
        if a == b:
            return True
        shorter, longer = sorted((a, b), key=len)
        return longer.startswith(shorter) and len(longer) - len(shorter) <= 3

    def _wup(self, a: str, b: str) -> float:
        wn = self._wn
        best = 0.0
        for sa in wn.synsets(a, pos=wn.NOUN)[:3]:
            for sb in wn.synsets(b, pos=wn.NOUN)[:3]:
                v = sa.wup_similarity(sb)
                if v and v > best:
                    best = v
        return best

    def _semantically_related(self, target: str, cand: str, cosine: float) -> bool:
        if cosine >= self.cfg.semantic_high_similarity:
            return True
        return cosine >= self.cfg.semantic_min_similarity and self._wup(target, cand) >= self.cfg.wordnet_min_wup

    # ---------------------------------------------------------------- candidate sets
    def semantic_candidates(self, target: str) -> tuple[list[int], np.ndarray]:
        if target in self._sem_cache:
            return self._sem_cache[target]
        lex = self.lexicon
        sim = lex.similarity(target); tp = lex.pron(target)
        out, w = [], []
        for i in np.argsort(-sim)[1: 200]:
            if sim[i] < self.cfg.semantic_min_similarity:
                break
            cand = lex.words[i]
            if self._morph_variant(cand, target) or formally_related(tp, lex.prons[i]):
                continue
            if self._semantically_related(target, cand, float(sim[i])):
                out.append(int(i)); w.append(float(sim[i]))
            if len(out) >= self.cfg.semantic_top_k:
                break
        self._sem_cache[target] = (out, np.array(w))
        return self._sem_cache[target]

    def formal_candidates(self, target: str) -> tuple[list[int], np.ndarray]:
        if target in self._form_cache:
            return self._form_cache[target]
        lex = self.lexicon
        tp = lex.pron(target); sim = lex.similarity(target)
        scored = []
        for i, p in enumerate(lex.prons):
            if abs(len(p) - len(tp)) > 2 or not formally_related(tp, p):
                continue
            cand = lex.words[i]
            if self._morph_variant(cand, target) or self._semantically_related(target, cand, float(sim[i])):
                continue
            scored.append((phoneme_edit_distance(tp, p), i))
        scored.sort()
        idx = [i for _, i in scored[: self.cfg.formal_top_k]]
        w = np.array([1.0 / (1 + d) for d, _ in scored[: self.cfg.formal_top_k]])
        self._form_cache[target] = (idx, w)
        return self._form_cache[target]

    def mixed_candidates(self, target: str) -> tuple[list[int], np.ndarray]:
        if target in self._mixed_cache:
            return self._mixed_cache[target]
        lex = self.lexicon
        tp = lex.pron(target); sim = lex.similarity(target)
        out, w = [], []
        for i in np.argsort(-sim)[1: 400]:
            if sim[i] < self.cfg.semantic_min_similarity:
                break
            cp = lex.prons[i]; cand = lex.words[i]
            if self._morph_variant(cand, target) or not formally_related(tp, cp):
                continue
            if cp[0] != tp[0] and phoneme_edit_distance(tp, cp) > 2:
                continue
            if self._semantically_related(target, cand, float(sim[i])):
                out.append(int(i)); w.append(float(sim[i]))
        self._mixed_cache[target] = (out, np.array(w))
        return self._mixed_cache[target]

    def unrelated_candidates(self, target: str) -> list[int]:
        if target in self._unrel_cache:
            return self._unrel_cache[target]
        lex = self.lexicon
        tp = lex.pron(target); sim = lex.similarity(target)
        pool = [int(i) for i in np.argsort(-lex.log_freq)[: self.cfg.unrelated_pool]
                if sim[i] < self.cfg.semantic_min_similarity and not formally_related(tp, lex.prons[i])
                and not self._morph_variant(lex.words[i], target)]
        self._unrel_cache[target] = pool
        return pool

    def _pick(self, idx: list[int], w: np.ndarray, rng: np.random.Generator) -> int:
        weights = w * np.exp(self.cfg.frequency_weight * self.lexicon.log_freq[idx])
        return idx[rng.choice(len(idx), p=weights / weights.sum())]

    # ---------------------------------------------------------------- nonwords
    def nonword(self, target: str, rng: np.random.Generator) -> tuple[str, ...]:
        """Phoneme edits of the target that are not any English word and are phonotactically legal.
        Edits are mostly within-class substitutions (vowel for vowel, consonant for consonant);
        deletions and insertions are rarer and never applied to very short targets."""
        lex = self.lexicon
        tp = tuple(lex.pron(target))
        for _ in range(200):
            n_edits = max(1, int(rng.binomial(len(tp), self.cfg.nonword_edit_rate)))
            out = list(tp)
            for _ in range(n_edits):
                if len(out) > 3 and rng.random() < self.cfg.nonword_indel_prob:
                    k = int(rng.integers(len(out)))
                    if rng.random() < 0.5:
                        del out[k]
                    else:
                        out.insert(k, str(rng.choice(CONSONANTS)))
                else:
                    k = int(rng.integers(len(out)))
                    pool = sorted(VOWELS) if out[k] in VOWELS else list(CONSONANTS)
                    out[k] = str(rng.choice([x for x in pool if x != out[k]]))
            res = tuple(out)
            if res != tp and not lex.is_word(res) and is_phonotactically_legal(res, reference=tp):
                return res
        # Deterministic last resort: substitute the first consonant with one that keeps legality.
        for c in CONSONANTS:
            out = list(tp)
            k = next((i for i, x in enumerate(out) if x not in VOWELS), 0)
            out[k] = c
            res = tuple(out)
            if res != tp and not lex.is_word(res) and is_phonotactically_legal(res, reference=tp):
                return res
        raise RuntimeError(f"could not build a legal nonword for {target!r}")

    # ---------------------------------------------------------------- main entry
    def realize(self, target: str, category: str, rng: np.random.Generator) -> Realization:
        lex = self.lexicon
        if lex.pron(target) is None:
            raise KeyError(f"{target!r} is not in the lexicon")
        original = category
        if category == "correct":
            return Realization("correct", target, lex.pron(target))
        if category == "omission":
            return Realization("omission", None, None)
        if category == "nonword":
            ph = self.nonword(target, rng)
            return Realization("nonword", "".join(p.lower() for p in ph), ph)
        idx: list[int] = []; w = np.array([])
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
                ph = self.nonword(target, rng)
                return Realization("nonword", "".join(p.lower() for p in ph), ph, fallback_from=original)
        if category == "unrelated":
            idx = self.unrelated_candidates(target); w = np.ones(len(idx))
        i = self._pick(idx, w, rng)
        return Realization(category, lex.words[i], lex.prons[i], fallback_from=None if category == original else original)
