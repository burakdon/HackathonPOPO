"""Automatic answer scoring (speech/scoring.py): the rules decided in docs/OPEN_DECISIONS.md D4-D6."""
from __future__ import annotations

import pytest

pytest.importorskip("cmudict")

from speech.base import Transcription, WordProb
from speech.scoring import SpeechUnavailable, content_words, score


def T(text: str, conf: float | None = 0.9, nsp: float | None = 0.02, status: str = "ok", **kw) -> Transcription:
    words = [WordProb(word=w, probability=conf) for w in text.split()] if conf is not None else []
    return Transcription(status=status, text=text, provider="test", audio_seconds=1.2, no_speech_prob=nsp,
                         words=words, **kw)


def S(text, preferred="cat", alts=("kitty", "kitten"), **kw):
    return score(T(text, **kw), preferred, list(alts))


@pytest.mark.parametrize("text, matched", [
    ("cat", "cat"), ("Cat.", "cat"), ("a cat", "cat"), ("uuummm cat", "cat"), ("Um, cat.", "cat"),
    ("hmm... uhh cat", "cat"), ("cat cat", "cat"), ("Kat.", "cat"), ("cats", "cat"), ("kitty", "kitty"), ("kitties", "kitty"), ("kitten", "kitten"),
])
def test_correct(text, matched):
    s = S(text)
    assert (s.outcome, s.matched_name, s.error_type) == ("correct", matched, None)


def test_same_sound_different_spelling_counts():
    s = score(T("pair"), "pear", [])
    assert s.outcome == "correct" and "sounds the same" in s.reason


@pytest.mark.parametrize("text", ["it's a cat", "cat no dog", "dog cat", "that's a cat", "big cat"])
def test_more_than_one_word_is_unclear(text):
    s = S(text)
    assert s.outcome == "unclear" and s.error_type is None and "More than one word" in s.reason


@pytest.mark.parametrize("text, heard", [("tat", "tat"), ("bat", "bat"), ("cap", "cap")])
def test_sound_errors(text, heard):
    s = S(text)
    assert (s.outcome, s.error_type, s.heard_word) == ("incorrect", "sound", heard)


@pytest.mark.parametrize("text, preferred", [("free", "tree"), ("tee", "tree")])
def test_sound_errors_other_targets(text, preferred):
    assert score(T(text), preferred, []).error_type == "sound"


@pytest.mark.parametrize("text, preferred", [("dog", "cat"), ("bone", "dog"), ("truck", "car"), ("bus", "car")])
def test_wrong_word(text, preferred):
    s = score(T(text), preferred, [])
    assert (s.outcome, s.error_type) == ("incorrect", "meaning")


def test_non_dictionary_words():
    assert S("cag").error_type == "sound"                    # not in CMUdict, spelled close to "cat"
    assert S("zzzglorp").error_type == "unclassified"
    # Known limit: spelling can't see that "kak" sounds like "cat" (k-a-k vs c-a-t: 2 of 3 letters differ)
    assert S("kak").error_type == "unclassified"


@pytest.mark.parametrize("text", ["I don't know", "Um, I don't know.", "dunno", "I'm not sure"])
def test_dont_know_is_no_response(text):
    s = S(text)
    assert (s.outcome, s.error_type) == ("incorrect", "no_response")


def test_unclear_cases():
    assert S("uuummm").outcome == "unclear"                                       # only fillers
    assert S("cat", conf=0.05).outcome == "unclear"                               # low confidence
    assert S("cat", conf=0.2).outcome == "correct"                                # Whisper is unsure of single words
    assert S("cat", nsp=0.9).outcome == "unclear"                                 # probably silence
    assert score(Transcription(status="empty", provider="t"), "cat", []).outcome == "unclear"
    assert score(Transcription(status="timeout", provider="t", error="slow"), "cat", []).outcome == "unclear"
    assert S("cat", conf=None).outcome == "correct"                               # no word confidences: not checked


def test_unavailable_raises():
    with pytest.raises(SpeechUnavailable):
        score(Transcription(status="unavailable", provider="t", error="not installed"), "cat", [])


def test_transcript_is_never_rewritten():
    s = S("tat")
    assert s.heard_word == "tat" and s.matched_name is None


def test_word_confidence_is_reported():
    assert S("cat", conf=0.83).signals.word_confidence == 0.83


def test_fillers_and_articles_only():
    assert content_words("Uuummm, the... uhh a CAT!") == ["cat"]
    assert content_words("ummm hmm mm erm") == []


def test_every_picture_name_scores_correct():
    from core.practice import load_items
    for it in load_items().items:
        for name in [it.preferred_name, *it.accepted_alternatives]:
            assert score(T(name), it.preferred_name, it.accepted_alternatives).outcome == "correct", name
