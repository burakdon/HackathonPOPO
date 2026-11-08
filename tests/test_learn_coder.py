"""The learned coder's feature extractor: fixed length, finite, deterministic, sensitive to the pair."""
import math

import pytest

from scripts.code_psst_responses import Coder
from scripts.learn_coder import FEATURE_NAMES, features


@pytest.fixture(scope="module")
def coder():
    return Coder()


def row(prompt, transcript, category="nonword_phon", confidence="medium"):
    return {"prompt": prompt, "transcript": transcript, "category": category, "confidence": confidence}


def test_feature_vector_shape_and_finiteness(coder):
    v = features(coder, row("beaver", "B IY V"))
    assert len(v) == len(FEATURE_NAMES)
    assert all(math.isfinite(x) for x in v)


def test_prefix_fragment_and_edit_features(coder):
    v = dict(zip(FEATURE_NAMES, features(coder, row("stethoscope", "S T EH"))))
    assert v["is_prefix_fragment"] == 1.0 and v["common_prefix"] == 3 and v["len_diff"] < 0
    w = dict(zip(FEATURE_NAMES, features(coder, row("stethoscope", "S T EH TH AH S K OW P"))))
    assert w["edit_distance"] == 0 and w["is_prefix_fragment"] == 0.0


def test_machine_code_is_one_hot(coder):
    v = dict(zip(FEATURE_NAMES, features(coder, row("beaver", "B IY V", category="formal", confidence="high"))))
    assert v["machine_formal"] == 1.0 and v["machine_nonword_phon"] == 0.0 and v["machine_confidence"] == 2
