"""Shared fixtures. The canonical example state is copied verbatim from the interface contract in the project spec."""
import pytest

from core.config import load_beta_bandit, load_clinical

EXAMPLE_STATE = {
    "last_trials": [{"word": "cat", "tier": 2, "cue": "first_sound", "outcome": False,
                     "error_type": "sound", "response_time": 3.1}],
    "current_tier": 2, "trial_count": 14, "session_minutes": 6.5,
}


def trial(outcome: bool = True, tier: int = 2, cue: str = "none", word: str = "cat") -> dict:
    return {"word": word, "tier": tier, "cue": cue, "outcome": outcome,
            "error_type": "correct" if outcome else "no_response", "response_time": 2.0}


@pytest.fixture
def clinical():
    return load_clinical()


@pytest.fixture
def bandit_cfg():
    return load_beta_bandit()
