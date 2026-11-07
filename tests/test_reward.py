"""Proxy reward: each term fires when it should, and tier oscillation cannot farm reward."""
import pytest

from core.reward import ProxyReward, load_reward


def tr(outcome, tier=2, cue="none"):
    return {"outcome": outcome, "tier": tier, "cue": cue}


@pytest.fixture
def reward(clinical):
    return ProxyReward(load_reward(), clinical)


def test_band_term_needs_a_full_window_and_in_band_accuracy(reward):
    seven_of_eight = [tr(True)] * 7 + [tr(False)]
    assert reward.step(seven_of_eight[:7])["band"] == 0.0           # window not full yet
    assert reward.step(seven_of_eight)["band"] == 1.0              # 7/8 = 0.875 in [0.75, 0.90]
    assert reward.step([tr(True)] * 8)["band"] == 0.0              # 8/8 = 1.0 above band
    assert reward.step([tr(True)] * 5 + [tr(False)] * 3)["band"] == 0.0   # 5/8 below band


def test_frustration_from_third_consecutive_failure(reward):
    assert reward.step([tr(True), tr(False), tr(False)])["frustration"] == 0.0
    assert reward.step([tr(True), tr(False), tr(False), tr(False)])["frustration"] == -1.0
    assert reward.step([tr(False)] * 5)["frustration"] == -1.0


def test_cue_waste_after_heavy_cue(reward):
    assert reward.step([tr(True, cue="first_sound"), tr(True)])["cue_waste"] == -0.1
    assert reward.step([tr(True, cue="first_letter"), tr(True)])["cue_waste"] == 0.0   # light cue
    assert reward.step([tr(False, cue="first_sound"), tr(True)])["cue_waste"] == 0.0   # cue did not help


def test_climb_bonus_only_at_end_and_only_if_session_stayed_in_band(reward):
    in_band = [tr(i % 8 != 0, tier=1 if i < 20 else 2) for i in range(40)]      # 7/8 correct throughout
    assert reward.terminal(in_band, start_tier=1)["climb"] == 2.0
    out_of_band = [tr(True, tier=1 if i < 20 else 3) for i in range(40)]         # 100% correct: never in band
    assert reward.terminal(out_of_band, start_tier=1)["climb"] == 0.0


def test_tier_oscillation_earns_nothing_per_step(reward):
    """The scaffold paid +reward per step for climbing while in band and charged nothing for dropping
    out of band, so oscillating tiers farmed reward. Here per-trial terms do not depend on the tier path:
    same outcomes and same start/end tier give the same total, however much the tier oscillates."""
    outcomes = [i % 8 != 0 for i in range(40)]
    hold = [tr(o, tier=2) for o in outcomes]
    oscillate = [tr(o, tier=1 if i % 2 else 2) for i, o in enumerate(outcomes)]
    oscillate[-1]["tier"] = 2
    assert reward.episode(oscillate, start_tier=1)["total"] == reward.episode(hold, start_tier=1)["total"]


@pytest.mark.parametrize("variant,active", [
    ("band_only", {"band"}), ("band_climb", {"band", "climb"}), ("full", {"band", "climb", "frustration", "cue_waste"}),
])
def test_ablation_variants_switch_terms(clinical, variant, active):
    r = ProxyReward(load_reward(variant=variant), clinical)
    trace = [tr(True, cue="first_sound"), tr(True)] + [tr(False)] * 3 + [tr(i % 8 != 0, tier=3) for i in range(35)]
    ep = r.episode(trace, start_tier=1)
    for term in ("band", "climb", "frustration", "cue_waste"):
        assert (ep[term] != 0.0) == (term in active), (variant, term, ep)


def test_probe_bonus_only_for_correct_probes_in_probe_variant(clinical):
    with_probe = ProxyReward(load_reward(variant="full_probe"), clinical)
    without = ProxyReward(load_reward(variant="full"), clinical)
    probe_ok = {"outcome": True, "tier": 2, "cue": "none", "probe": True}
    probe_fail = {"outcome": False, "tier": 2, "cue": "none", "probe": True}
    assert with_probe.step([tr(True), probe_ok])["probe"] == 1.0
    assert with_probe.step([tr(True), probe_fail])["probe"] == 0.0
    assert without.step([tr(True), probe_ok])["probe"] == 0.0
