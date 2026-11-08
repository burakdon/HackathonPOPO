"""The simulated cohorts must reproduce published group-level picture-naming statistics
(data/calibration_targets.yaml: Kittredge et al. 2008, Dell et al. 2007, Meteyard & Bose 2018,
Mirman et al. 2024). One test per target so a regression names the statistic that broke."""
import pytest

from core.config import load_yaml
from eval.calibration import check, measure

TARGETS = load_yaml("data/calibration_targets.yaml")


@pytest.fixture(scope="module")
def results():
    return {r["target"]: r for r in check(measure(), TARGETS)}


@pytest.mark.parametrize("target", list(TARGETS["targets"]))
def test_simulator_matches_published_statistic(results, target):
    r = results[target]
    assert r["pass"], f"{target}: simulator {r['value']:.3f} outside published range [{r['low']}, {r['high']}] ({r['source']})"


def test_every_target_cites_a_source_and_a_measure():
    for key, t in TARGETS["targets"].items():
        srcs = [t["source"]] if isinstance(t["source"], str) else t["source"]
        assert all(s in TARGETS["sources"] for s in srcs), key
        assert t["measure"] and len(t["range"]) == 2, key
