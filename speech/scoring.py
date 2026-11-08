"""Automatic scoring of one picture-naming answer (docs/OPEN_DECISIONS.md D4-D6, API contract §4).

Fixed rules, no learning, applied to the recognizer's transcript, which is never rewritten:

1. Unclear (asks for another recording, never counted): no speech, a failed or timed-out
   transcription, the recognizer reporting probable silence, low confidence in the word, no word
   left after fillers, or two or more words ("it's a cat").
2. Fillers are removed first, however drawn out ("uuummm", "uhh", "hmm") together with "a", "an",
   "the". So "uuummm cat" is the single word "cat". A word repeated straight away counts once
   ("cat cat"; the recognizer also produces such repeats on its own).
3. "I don't know" and similar phrases are an incorrect answer of type no_response.
4. Correct: the single word is the expected name or an accepted alternative, its plural, or a word
   that sounds exactly the same in the CMU Pronouncing Dictionary ("pair" for "pear").
5. Incorrect otherwise. Error type: `sound` when the word sounds close to an accepted name (phoneme
   edit distance at most half the length; spelling distance for words the dictionary doesn't
   know), `meaning` when it is a different real word, `unclassified` when it is neither.

The thresholds are first guesses, to be checked on the synthetic clips (D9). The error type is
approximate: the recognizer tends to turn near-misses into real words.
"""
from __future__ import annotations

import re
from functools import lru_cache

from core.contract import Score, Signals
from speech.base import Transcription

# Recognizer confidence in the single word, below this: unclear. Whisper is unsure of isolated words
# even when right (Mac-voice check, 2026-11-07: "cat" 0.41, "dog" 0.17, "free" 0.29), so this is low.
MIN_WORD_CONFIDENCE = 0.1
MAX_NO_SPEECH_PROB = 0.6        # recognizer's own "this was probably silence" estimate
SOUND_DISTANCE = 0.5            # normalized edit distance at or below this: a sound error

_FILLER = re.compile(r"^(u+h*m+|u+h+|e+r+m*|e+h+|h+m+|m+h*m*|a+h+|o+h+)$")   # um, uuummm, uhh, er(m), hmm, mm, ah, oh
_ARTICLES = {"a", "an", "the"}
_DONT_KNOW = {"i don't know", "i dont know", "don't know", "dont know", "dunno", "i dunno", "no idea",
              "i have no idea", "i'm not sure", "im not sure", "not sure", "i can't", "i cant", "i forget",
              "i forgot", "pass", "skip"}


class SpeechUnavailable(RuntimeError):
    """No speech recognizer is configured: the endpoint answers 503 and records nothing."""


@lru_cache(maxsize=1)
def _cmu() -> dict[str, list[list[str]]]:
    import cmudict
    return cmudict.dict()


def pronunciations(word: str) -> list[tuple[str, ...]]:
    """CMUdict pronunciations with stress marks removed; [] for words the dictionary doesn't know."""
    return [tuple(re.sub(r"\d", "", p) for p in pron) for pron in _cmu().get(word.lower(), [])]


def tokens(text: str) -> list[str]:
    t = text.lower().replace("’", "'")
    t = re.sub(r"[^a-z' ]+", " ", t)
    return [w.strip("'") for w in t.split() if w.strip("'")]


def content_words(text: str) -> list[str]:
    """Words left after fillers and articles; a word repeated straight away ("cat cat") counts once."""
    out: list[str] = []
    for w in tokens(text):
        if not _FILLER.match(w) and w not in _ARTICLES and (not out or out[-1] != w):
            out.append(w)
    return out


def _plurals(name: str) -> set[str]:
    out = {name + "s", name + "es"}
    if name.endswith("y"):
        out.add(name[:-1] + "ies")
    return out


def _edit(a, b) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i]
        for j, y in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y)))
        prev = cur
    return prev[-1]


def _norm_distance(a, b) -> float:
    return _edit(a, b) / max(len(a), len(b), 1)


def _quote(s: str) -> str:
    return f"“{s}”"


def score(t: Transcription, preferred: str, alternatives: list[str]) -> Score:
    sig = Signals(audio_seconds=t.audio_seconds, no_speech_prob=t.no_speech_prob, word_confidence=None)

    def unclear(reason: str, word_conf: float | None = None) -> Score:
        s = sig.model_copy(update={"word_confidence": word_conf})
        return Score(outcome="unclear", error_type=None, heard_word=None, matched_name=None, reason=reason, signals=s)

    if t.status == "unavailable":
        raise SpeechUnavailable(t.error or "speech recognition unavailable")
    if t.status == "empty":
        return unclear("No speech detected.")
    if t.status in ("failed", "timeout"):
        return unclear(f"The recording could not be processed ({t.error or t.status}).")
    if t.no_speech_prob is not None and t.no_speech_prob > MAX_NO_SPEECH_PROB:
        return unclear("The recognizer judged the recording to be mostly silence.")

    words = content_words(t.text)
    phrase = " ".join(tokens(t.text))
    if phrase in _DONT_KNOW or " ".join(words) in _DONT_KNOW:
        return Score(outcome="incorrect", error_type="no_response", heard_word=None, matched_name=None,
                     reason=f"The answer {_quote(t.text.strip())} means “I don't know”.", signals=sig)
    if not words:
        return unclear(f"Only filler sounds ({_quote(t.text.strip())}); no word to score.")
    if len(words) > 1:
        return unclear(f"More than one word ({_quote(t.text.strip())}); only single-word answers are scored.")

    word = words[0]
    conf = max((w.probability for w in t.words if content_words(w.word) == [word]), default=None)
    sig = sig.model_copy(update={"word_confidence": conf})
    if conf is not None and conf < MIN_WORD_CONFIDENCE:
        return unclear(f"The recognizer wasn't confident it heard {_quote(word)} ({conf:.2f}).", conf)

    names = [n.lower() for n in (preferred, *alternatives)]
    after_fillers = "" if tokens(t.text) == [word] else "After removing fillers and repeats, "
    lead = f"{after_fillers}one word, {_quote(word)}" if after_fillers else f"One word, {_quote(word)}"

    def correct(name: str, how: str) -> Score:
        return Score(outcome="correct", error_type=None, heard_word=word, matched_name=name,
                     reason=f"{lead}, {how}.", signals=sig)

    for n in names:
        if word == n:
            return correct(n, f"matches {_quote(n)}")
    for n in names:
        if word in _plurals(n):
            return correct(n, f"the plural of {_quote(n)}")
    heard = pronunciations(word)
    for n in names:
        if heard and set(heard) & set(pronunciations(n)):
            return correct(n, f"sounds the same as {_quote(n)}")

    def incorrect(error: str, why: str) -> Score:
        return Score(outcome="incorrect", error_type=error, heard_word=word, matched_name=None,
                     reason=f"{lead}, {why}.", signals=sig)

    target = _quote(preferred.lower())
    if heard:
        d = min(_norm_distance(h, p) for h in heard for n in names for p in pronunciations(n) or [tuple(n)])
        if d <= SOUND_DISTANCE:
            return incorrect("sound", f"sounds close to {target}")
        return incorrect("meaning", f"a different word from {target}")
    d = min(_norm_distance(word, n) for n in names)
    if d <= SOUND_DISTANCE:
        return incorrect("sound", f"spelled close to {target} (not a dictionary word)")
    return incorrect("unclassified", f"not a dictionary word and not close to {target}")
