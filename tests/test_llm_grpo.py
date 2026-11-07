"""LLM prompt/parse and the GRPO reward and dataset (offline: no API calls)."""
import numpy as np
import pytest

from agents.llm_prompt import messages, parse_action
from core.config import load_clinical
from core.masks import ACTIONS, action_index, action_mask
from core.schema import State
from train.grpo_data import build_dataset, reward


@pytest.mark.parametrize("text,expected", [
    ('{"tier": 2, "cue": "none"}', (2, "none")),
    ('Sure! {"tier": 3, "cue": "first_sound", "rationale": "sound errors"} done', (3, "first_sound")),
    ('{"tier": 4, "cue": "none"} {"tier": 1, "cue": "semantic"}', (1, "semantic")),   # first VALID object
    ('{"tier": "2", "cue": "none"}', None),
    ('tier 2 with no cue', None),
    ('{"tier": 2, "cue": "gesture"}', None),
])
def test_parse_action(text, expected):
    assert parse_action(text) == expected


def test_prompt_lists_only_allowed_actions():
    state = State(current_tier=1, trial_count=0, session_minutes=0.0)
    mask = action_mask(1, ["none", "semantic"], 1)
    user = messages(state, mask)[1]["content"]
    assert '"tier":3' not in user and '{"tier":2,"cue":"semantic"}' in user and "first_sound" not in user.split("Allowed")[1]


@pytest.fixture(scope="module")
def dataset():
    return build_dataset(40, seed=3)


def test_dataset_is_deterministic_and_well_formed(dataset):
    again = build_dataset(40, seed=3)
    assert [d["p_true"] for d in dataset] == [d["p_true"] for d in again]
    for d in dataset:
        assert len(d["p_true"]) == 15 and len(d["mask"]) == 15 and any(d["mask"])
        State(**d["state"])


def test_reward_prefers_on_target_action_and_penalises_invalid(dataset):
    clinical = load_clinical()
    ex = dataset[0]
    allowed = [i for i in range(15) if ex["mask"][i]]
    best = min(allowed, key=lambda i: abs(ex["p_true"][i] - clinical.target))
    worst = max(allowed, key=lambda i: abs(ex["p_true"][i] - clinical.target))
    as_json = lambda i: '{"tier": %d, "cue": "%s"}' % ACTIONS[i]
    r = lambda txt: reward(txt, ex, clinical, cue_cost=0.0, tier_bonus=0.0, format_penalty=-1.0)
    assert r(as_json(best)) >= r(as_json(worst))
    assert r("no json here") == -1.0
    masked = next(i for i in range(15) if not ex["mask"][i])
    assert r(as_json(masked)) == -1.0
