"""The automatic PSST response coder on hand-built cases."""
from __future__ import annotations

import pytest

from scripts.code_psst_responses import Coder


@pytest.fixture(scope="module")
def coder() -> Coder:
    return Coder()


def row(prompt: str, transcript: str, test: str = "BNT", correct: bool = False) -> dict:
    return {"utterance_id": f"x-{prompt}", "session": "S", "test": test, "split": "train", "aq_index": "70",
            "prompt": prompt, "correctness": "True" if correct else "False", "transcript": transcript}


def test_correct_and_variants(coder: Coder) -> None:
    assert coder.code(row("house", "HH AW S", correct=True))["category"] == "correct"
    out = coder.code(row("comb", "K OW M Z"))
    assert out["category"] == "correct_variant" and out["confidence"] == "high"


def test_single_word_errors(coder: Coder) -> None:
    assert coder.code(row("canoe", "B OW T"))["category"] == "semantic"           # boat
    assert coder.code(row("octopus", "IY L"))["category"] == "semantic"           # eel
    out = coder.code(row("bench", "CH EY R"))                                      # chair, one vowel off
    assert out["category"] == "semantic" and out["matched_word"] == "chair" and out["confidence"] == "medium"
    out = coder.code(row("toothbrush", "B R AH SH"))                               # brush: related and shares sounds
    assert out["category"] == "mixed"
    assert coder.code(row("cat", "M AE T"))["category"] == "formal"                # mat
    assert coder.code(row("house", "T EY B AH L"))["category"] == "unrelated"      # table


def test_nonwords_fragments_and_multiword(coder: Coder) -> None:
    out = coder.code(row("stethoscope", "S T EH TH AH S OW P"))                    # one edit from target
    assert out["category"] == "nonword_phon" and out["confidence"] == "high"
    assert coder.code(row("canoe", "G AH N UW"))["category"] == "nonword_phon"
    assert coder.code(row("stethoscope", "S T EH"))["category"] == "fragment"
    assert coder.code(row("sphinx", "CH IH P SH UW N"))["category"] == "nonword_other"
    out = coder.code(row("unicorn", "IH T S AH HH AO R S B AH T IH T S AH HH AO R S W IH DH EY"))
    assert out["category"] == "multiword"


def test_filled_pauses_are_stripped(coder: Coder) -> None:
    assert coder.code(row("unicorn", "AH HH AO R S"))["matched_word"] == "horse"


def test_proper_nouns_do_not_count_as_words(coder: Coder) -> None:
    out = coder.code(row("beaver", "B IY B ER"))                                   # "bieber" is not a word here
    assert out["category"] == "nonword_phon"
