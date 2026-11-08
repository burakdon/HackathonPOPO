"""Conservative transcript normalisation and a *suggested* match against the picture's names.

The suggestion pre-selects "correct" for the helper when the target word clearly appears; it never
decides anything on its own, and it is never evidence of an error type (a mismatch can be a
recognition error, a different accepted name, or a real naming error).
"""
from __future__ import annotations

import re

from pydantic import BaseModel

FILLERS = {"um", "uh", "er", "erm", "hmm", "mm", "ah", "oh", "a", "an", "the", "it's", "its", "it", "is",
           "that's", "thats", "that", "this", "like", "uhm"}


def normalize(text: str) -> list[str]:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z' ]+", " ", t)
    return [w for w in t.split() if w and w not in FILLERS]


def _variants(word: str) -> set[str]:
    w = word.lower()
    out = {w}
    if w.endswith("s") and len(w) > 3:
        out.add(w[:-1])
    else:
        out.add(w + "s")
    return out


class MatchSuggestion(BaseModel):
    suggestion: str            # "matches" | "no_match" | "empty"
    matched_name: str | None = None
    normalized: list[str]
    note: str


def suggest_match(transcript: str, preferred: str, alternatives: list[str]) -> MatchSuggestion:
    words = normalize(transcript)
    if not words:
        return MatchSuggestion(suggestion="empty", normalized=words, note="nothing to compare")
    names = [preferred, *alternatives]
    joined = " ".join(words)
    for name in names:
        n = " ".join(normalize(name)) or name.lower()
        if " " in n:
            if n in joined:
                return MatchSuggestion(suggestion="matches", matched_name=name, normalized=words,
                                       note="suggestion only: the helper confirms")
        elif any(w in _variants(n) for w in words):
            return MatchSuggestion(suggestion="matches", matched_name=name, normalized=words,
                                   note="suggestion only: the helper confirms")
    return MatchSuggestion(suggestion="no_match", normalized=words,
                           note="suggestion only: may be a recognition error or another valid name")
