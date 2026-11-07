"""Action masks encode the hard clinical constraints."""
import numpy as np

from core.masks import ACTIONS, N_ACTIONS, action_index, action_mask
from core.schema import CUES, TIERS


def test_action_space_is_15_tier_major():
    assert N_ACTIONS == 15
    for i, (t, c) in enumerate(ACTIONS):
        assert action_index(t, c) == i


def test_tier_step_constraint():
    for current in TIERS:
        m = action_mask(current, CUES, max_tier_step=1)
        for i, (t, _) in enumerate(ACTIONS):
            assert m[i] == (abs(t - current) <= 1)


def test_unapproved_cues_masked():
    m = action_mask(2, ["none", "semantic"], max_tier_step=1)
    allowed = {ACTIONS[i] for i in np.flatnonzero(m)}
    assert allowed == {(t, c) for t in (1, 2, 3) for c in ("none", "semantic")}
