"""A lexicon with pronunciations, frequencies, semantic embeddings and WordNet nouns for realising
errors as words.

Vocabulary: the IPNP item names plus concrete English words from the Brysbaert, Warriner & Kuperman
(2014) concreteness norms (mean rating >= `concreteness_min` on a 1-5 scale, known by >= 90% of
raters) that have a CMUdict pronunciation, are nouns in WordNet (more noun senses than verb or
adjective senses), and are not on a profanity list. Picture-naming errors are overwhelmingly
picturable nouns. Falls back to the most frequent wordfreq words when the norms file is absent.

Pronunciations are stress-stripped ARPAbet tuples, the alphabet of the PSST transcripts. The full
CMUdict pronunciation set is kept so that nonwords can be checked against all of English, not just
this vocabulary. Embeddings come from a small sentence-transformers model and are cached on disk.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = REPO_ROOT / "data" / "cache"
NLTK_DATA = CACHE_DIR / "nltk_data"
BRYSBAERT_XLSX = REPO_ROOT / "data" / "raw" / "brysbaert" / "brysbaert2014_concreteness.xlsx"

VOWELS = frozenset({"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY", "IH", "IY", "OW", "OY", "UH", "UW"})
CONSONANTS = ("B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L", "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V", "W", "Y", "Z", "ZH")
# English phonotactics used to keep generated nonwords pronounceable.
PREVOCALIC_ONLY = frozenset({"HH", "W", "Y"})      # never syllable-final
# Words that pass the concreteness, noun and profanity filters but should never be produced as a
# patient's error in a clinical or demo setting (sexual, bodily, identity or violence terms).
SENSITIVE_WORDS = frozenset({
    "lesbian", "gay", "homosexual", "butt", "buttock", "buttocks", "boob", "breast", "breasts", "nipple", "penis",
    "vagina", "vulva", "testicle", "testicles", "scrotum", "anus", "rectum", "genital", "genitals", "condom", "tampon",
    "corpse", "cadaver", "coffin", "casket", "gun", "pistol", "rifle", "shotgun", "bullet", "grenade", "bomb", "knife",
    "dagger", "noose", "gallows", "slave", "negro", "nazi", "whore", "prostitute", "stripper", "drug", "cocaine",
    "heroin", "marijuana", "tumor", "cancer",
})
NEVER_INITIAL = frozenset({"NG", "ZH"})
LEGAL_ONSET_CLUSTERS = frozenset({
    ("P", "L"), ("P", "R"), ("B", "L"), ("B", "R"), ("T", "R"), ("T", "W"), ("D", "R"), ("D", "W"), ("K", "L"), ("K", "R"),
    ("K", "W"), ("G", "L"), ("G", "R"), ("G", "W"), ("F", "L"), ("F", "R"), ("TH", "R"), ("TH", "W"), ("SH", "R"), ("S", "P"),
    ("S", "T"), ("S", "K"), ("S", "M"), ("S", "N"), ("S", "L"), ("S", "W"), ("S", "F"), ("P", "Y"), ("B", "Y"), ("K", "Y"),
    ("G", "Y"), ("F", "Y"), ("V", "Y"), ("M", "Y"), ("N", "Y"), ("HH", "Y"), ("S", "P", "L"), ("S", "P", "R"), ("S", "T", "R"),
    ("S", "K", "R"), ("S", "K", "W"),
})


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


def _consonant_runs(p: tuple[str, ...]) -> list[tuple[str, ...]]:
    runs, run = [], []
    for x in list(p) + ["#"]:
        if x in VOWELS or x == "#":
            if run:
                runs.append(tuple(run)); run = []
        else:
            run.append(x)
    return runs


def is_phonotactically_legal(p: tuple[str, ...], reference: tuple[str, ...] | None = None) -> bool:
    """Coarse English legality: at least one vowel; no identical adjacent phonemes; HH/W/Y only before
    a vowel; NG/ZH not word-initial; a word-initial consonant cluster must be a legal English onset;
    no consonant run longer than three. With `reference` (the target a nonword was derived from),
    clusters the target itself contains are accepted too, so real words with long medial clusters
    such as "corkscrew" (R K S K R) can still yield legal nonwords."""
    if not p or not any(x in VOWELS for x in p):
        return False
    if any(x == y for x, y in zip(p, p[1:])):
        return False
    if p[0] in NEVER_INITIAL:
        return False
    for i, x in enumerate(p):
        if x in PREVOCALIC_ONLY and (i + 1 >= len(p) or p[i + 1] not in VOWELS):
            return False
    runs = _consonant_runs(p)
    if not runs:
        return True
    ref_runs = set(_consonant_runs(reference)) if reference else set()
    max_run = max([3] + [len(r) for r in ref_runs])
    if p[0] not in VOWELS and len(runs[0]) > 1 and runs[0] not in LEGAL_ONSET_CLUSTERS and runs[0] not in ref_runs:
        return False
    return all(len(r) <= max_run or r in ref_runs for r in runs)


@dataclass
class Lexicon:
    words: list[str]
    prons: list[tuple[str, ...]]
    log_freq: np.ndarray                 # Zipf frequency (wordfreq), ~1-7
    embeddings: np.ndarray               # (N, d), L2-normalised
    index: dict[str, int]
    all_prons: frozenset                 # every stress-stripped pronunciation in CMUdict
    _by_pron: dict[tuple[str, ...], str]

    def __len__(self) -> int:
        return len(self.words)

    def pron(self, word: str) -> tuple[str, ...] | None:
        i = self.index.get(word)
        return self.prons[i] if i is not None else None

    def is_word(self, pron: tuple[str, ...]) -> bool:
        """True if this pronunciation is any English word in CMUdict (not just our vocabulary)."""
        return pron in self.all_prons

    def similarity(self, word: str) -> np.ndarray:
        return self.embeddings @ self.embeddings[self.index[word]]


def wordnet():
    import nltk
    if str(NLTK_DATA) not in nltk.data.path:
        nltk.data.path.insert(0, str(NLTK_DATA))
    try:
        from nltk.corpus import wordnet as wn
        wn.synsets("cat")
    except LookupError:
        nltk.download("wordnet", download_dir=str(NLTK_DATA), quiet=True)
        from nltk.corpus import wordnet as wn
    return wn


def is_noun(word: str, wn) -> bool:
    n = len(wn.synsets(word, pos=wn.NOUN))
    return n > 0 and n >= len(wn.synsets(word, pos=wn.VERB)) + len(wn.synsets(word, pos=wn.ADJ))


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
                  nouns_only: bool = True, model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                  cache: bool = True) -> Lexicon:
    import cmudict
    from wordfreq import zipf_frequency

    extra = [w.lower() for w in (extra_words or [])]
    cmu = cmudict.dict()
    vocab = [w for w in _default_vocabulary(extra, size, concreteness_min) if w in cmu]
    if nouns_only:
        from better_profanity import profanity
        wn = wordnet()
        keep = set(extra)
        vocab = [w for w in vocab if (w in keep or is_noun(w, wn)) and not profanity.contains_profanity(w)
                 and w not in SENSITIVE_WORDS]
    prons = [strip_stress(cmu[w][0]) for w in vocab]
    log_freq = np.array([zipf_frequency(w, "en") for w in vocab])
    all_prons = frozenset(strip_stress(p) for ps in cmu.values() for p in ps)

    key = hashlib.md5(("|".join(vocab) + model_name).encode()).hexdigest()[:12]
    cache_path = CACHE_DIR / f"lexicon_{len(vocab)}_{key}.npz"
    emb = None
    if cache and cache_path.exists():
        z = np.load(cache_path, allow_pickle=False)
        if list(z["words"]) == vocab:
            emb = z["embeddings"]
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
    return Lexicon(vocab, prons, log_freq, emb, {w: i for i, w in enumerate(vocab)}, all_prons, by_pron)
