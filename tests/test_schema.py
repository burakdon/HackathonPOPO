"""The State/Action/Outcome contract: what the frontend may send and what the agent returns."""
import pytest
from pydantic import ValidationError

from core.config import ClinicalConfig, load_clinical
from core.schema import Action, Outcome, State
from tests.conftest import EXAMPLE_STATE, trial


def test_contract_example_state_parses():
    s = State(**EXAMPLE_STATE)
    assert s.session_id is None
    assert s.last_trials[0].error_type == "sound"


def test_session_id_is_optional_and_roundtrips():
    s = State(**EXAMPLE_STATE, session_id="patient-42")
    assert State.model_validate_json(s.model_dump_json()).session_id == "patient-42"


def test_last_trials_truncated_to_ten_most_recent():
    trials = [trial(word=f"w{i}") for i in range(15)]
    s = State(last_trials=trials, current_tier=1, trial_count=15, session_minutes=3.0)
    assert [t.word for t in s.last_trials] == [f"w{i}" for i in range(5, 15)]


@pytest.mark.parametrize("bad", [
    {"current_tier": 0}, {"current_tier": 4}, {"trial_count": -1}, {"session_minutes": -0.1},
])
def test_state_rejects_out_of_range(bad):
    with pytest.raises(ValidationError):
        State(**{**EXAMPLE_STATE, **bad})


def test_state_rejects_unknown_cue_and_error_type():
    bad_cue = {**EXAMPLE_STATE, "last_trials": [{**EXAMPLE_STATE["last_trials"][0], "cue": "gesture"}]}
    bad_err = {**EXAMPLE_STATE, "last_trials": [{**EXAMPLE_STATE["last_trials"][0], "error_type": "other"}]}
    for bad in (bad_cue, bad_err):
        with pytest.raises(ValidationError):
            State(**bad)


@pytest.mark.parametrize("correct,error_type", [(True, "sound"), (False, "correct")])
def test_outcome_rejects_inconsistent_correct_and_error_type(correct, error_type):
    with pytest.raises(ValidationError):
        Outcome(correct=correct, error_type=error_type, response_time=1.0)
    t = {**EXAMPLE_STATE["last_trials"][0], "outcome": correct, "error_type": error_type}
    with pytest.raises(ValidationError):
        State(**{**EXAMPLE_STATE, "last_trials": [t]})


def test_extra_fields_ignored_for_forward_compatibility():
    s = State(**EXAMPLE_STATE, timestamp="2026-11-07T12:00:00Z")
    assert not hasattr(s, "timestamp")


def test_action_json_matches_contract_keys():
    a = Action(tier=2, cue="first_sound", rationale="short text")
    assert set(a.model_dump()) == {"tier", "cue", "rationale"}
    with pytest.raises(ValidationError):
        Action(tier=4, cue="none")
    with pytest.raises(ValidationError):
        Action(tier=1, cue="whisper")


def test_clinical_config_loads_and_validates():
    c = load_clinical()
    assert c.band == (0.75, 0.90) and c.target == 0.85 and c.window == 8
    with pytest.raises(ValidationError):  # 'none' must stay approved so masks are never empty
        ClinicalConfig(**{**c.model_dump(), "approved_cues": ["first_sound"]})
    with pytest.raises(ValidationError):  # target outside band
        ClinicalConfig(**{**c.model_dump(), "target": 0.95})
