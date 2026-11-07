"""The tier-1 coasting fix: on the mild preset the contextual bandit must progress past tier 1."""
import numpy as np

from agents.logistic_bandit import LogisticThompsonBandit
from core.config import load_clinical, load_logistic_bandit
from sim.config import load_sim
from sim.items import load_items
from sim.population import make_patients
from sim.session import run_session

FIRST_TIER2_BY = 15        # reach tier >= 2 within this many trials ...
MIN_SHARE_REACHING = 0.9   # ... in at least this share of sessions
MAX_MEAN_FRAC_TIER1 = 0.5  # and spend less than half the session on tier 1 on average


def test_mild_patients_progress_beyond_tier_one():
    cfg, clinical = load_sim(), load_clinical()
    bank, lcfg = load_items(cfg.items), load_logistic_bandit()
    first_tier2, frac_tier1 = [], []
    for spec in make_patients(cfg, "mild_anomic", "test", 50, root_seed=0):
        patient, selector = spec.build(bank, cfg, clinical)
        r = run_session(LogisticThompsonBandit(lcfg, clinical, spec.index), patient, selector, cfg, clinical)
        tiers = np.array([t["tier"] for t in r.trace])
        first_tier2.append(int(np.argmax(tiers >= 2)) if (tiers >= 2).any() else len(tiers))
        frac_tier1.append(float(np.mean(tiers == 1)))
    share = np.mean(np.array(first_tier2) < FIRST_TIER2_BY)
    print(f"reached tier 2 within {FIRST_TIER2_BY} trials in {share:.0%} of sessions; "
          f"median first tier-2 trial {np.median(first_tier2):.0f}; mean share on tier 1 {np.mean(frac_tier1):.2f}")
    assert share >= MIN_SHARE_REACHING
    assert np.mean(frac_tier1) < MAX_MEAN_FRAC_TIER1
