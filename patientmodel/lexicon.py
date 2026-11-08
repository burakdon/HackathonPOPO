"""A lexicon with pronunciations, frequencies and semantic embeddings for realising errors as words.

Vocabulary: the IPNP item names plus concrete English words from the Brysbaert, Warriner & Kuperman
(2014) concreteness norms (mean rating >= `concreteness_min` on a 1-5 scale, known by >= 90% of
raters) that have a CMUdict pronunciation. Picture-naming errors are overwhelmingly concrete nouns,
and concreteness is the best single-column proxy for that in a 40k-word list. Falls back to the most
frequent wordfreq words when the norms file is absent. Pronunciations are stress-stripped ARPAbet tuples, the same alphabet as the PSST
transcripts. Embeddings come from a small sentence-transformers model and are cached on disk.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"
BRYSBAERT_XLSX = REPO_ROOT / "data" / "raw" / "brysbaert" / "brysbaert2014_concreteness.xlsx"
VOWELS = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"}


def strip_stress(phones: list[str]) -> tuple[str, ...]:
    return tuple(re.sub(r"\d", "", p) for p in phones)


def phoneme_edit_distance(a: tuple[str, ...], b: tuple[str, ...]) -> int:
    D = np.zeros((len(a) + 1, len(b) + 1), dtype=int)
    D[:, 0] = np.arange(len(a) + 1); D[0, :] = np.arange(len(b) + 1)
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            D[i, j] = min(D[i - 1, j] + 1, D[i, j - 1] + 1, D[i - 1, j - 1] + (a[i - 1] != b[j - 1]))
    return int(D[-1, -1])


def formally_related(a: tuple[str, ...], b: tuple[str, ...]) -> bool:
    """PNT formal-relatedness criterion (Walker 2016, fn.): share the initial or final phoneme, or a
    phoneme in the same position aligned from the left, or any two phonemes in any position."""
    if not a or not b:
        return False
    if a[0] == b[0] or a[-1] == b[-1]:
        return True
    if any(x == y for x, y in zip(a, b)):
        return True
    return len(set(a) & set(b)) >= 2


@dataclass
class Lexicon:
    words: list[str]
    prons: list[tuple[str, ...]]
    log_freq: np.ndarray                 # Zipf frequency (wordfreq), ~1-7
    embeddings: np.ndarray               # (N, d), L2-normalised
    index: dict[str, int]
    _by_pron: dict[tuple[str, ...], str]

    def __len__(self) -> int:
        return len(self.words)

    def pron(self, word: str) -> tuple[str, ...] | None:
        i = self.index.get(word)
        return self.prons[i] if i is not None else None

    def is_word(self, pron: tuple[str, ...]) -> bool:
        return pron in self._by_pron

    def similarity(self, word: str) -> np.ndarray:
        return self.embeddings @ self.embeddings[self.index[word]]


def _concrete_vocabulary(concreteness_min: float, known_min: float = 0.9) -> list[str] | None:
    if not BRYSBAERT_XLSX.exists():
        return None
    import pandas as pd
    df = pd.read_excel(BRYSBAERT_XLSX)
    m = (df["Bigram"] == 0) & (df["Conc.M"] >= concreteness_min) & (df["Percent_known"] >= known_min)
    return [str(w).lower() for w in df.loc[m, "Word"] if str(w).isalpha() and len(str(w)) >= 3]


def _default_vocabulary(extra: list[str], size: int, concreteness_min: float) -> list[str]:
    words = _concrete_vocabulary(concreteness_min)
    if words is None:
        from wordfreq import top_n_list
        words = [w for w in top_n_list("en", size) if w.isalpha() and len(w) >= 3]
    seen, out = set(), []
    for w in list(extra) + words:
        if w not in seen:
            seen.add(w); out.append(w)
    return out


def build_lexicon(extra_words: list[str] | None = None, size: int = 20000, concreteness_min: float = 4.0,
                  model_name: str = "sentence-transformers/all-MiniLM-L6-v2", cache: bool = True) -> Lexicon:
    import cmudict
    from wordfreq import zipf_frequency

    extra = [w.lower() for w in (extra_words or [])]
    cmu = cmudict.dict()
    vocab = [w for w in _default_vocabulary(extra, size, concreteness_min) if w in cmu]
    import hashlib
    key = hashlib.md5(("|".join(vocab) + model_name).encode()).hexdigest()[:12]
    cache_path = CACHE_DIR / f"lexicon_{len(vocab)}_{key}.npz"
    prons = [strip_stress(cmu[w][0]) for w in vocab]
    log_freq = np.array([zipf_frequency(w, "en") for w in vocab])

    if cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if list(z["words"]) == vocab:
            emb = z["embeddings"]
        else:
            emb = None
    else:
        emb = None
    if emb is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        emb = model.encode(vocab, batch_size=256, normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        if cache:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache_path, words=np.array(vocab), embeddings=emb)

    by_pron: dict[tuple[str, ...], str] = {}
    for w, p in zip(vocab, prons):
        by_pron.setdefault(p, w)
    return Lexicon(vocab, prons, log_freq, emb, {w: i for i, w in enumerate(vocab)}, by_pron)
