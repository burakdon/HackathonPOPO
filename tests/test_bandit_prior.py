"""Empirical-Bayes prior: correct fitting, valid artifact, and a person's data still dominates it."""
import numpy as np

from agents.logistic_bandit import FEATURES, N_FEATURES, LogisticThompsonBandit
from core.config import load_logistic_bandit, load_yaml
from core.masks import action_index
from core.schema import Action, Outcome, State
from train.fit_bandit_prior import map_logistic


def test_map_logistic_recovers_known_weights():
    rng = np.random.default_rng(0)
    w_true = np.array([0.5, -1.0, 2.0])
    X = np.hstack([np.ones((4000, 1)), rng.normal(size=(4000, 2))])
    y = (rng.random(4000) < 1 / (1 + np.exp(-X @ w_true))).astype(float)
    w = map_logistic(X, y, np.zeros(3), np.eye(3) * 1e-3, iters=50)
    assert np.allclose(w, w_true, atol=0.15)


def test_fitted_prior_artifact_is_valid():
    data = load_yaml(load_logistic_bandit().prior_file)
    assert data["features"] == FEATURES
    cov = np.array(data["cov"])
    assert cov.shape == (N_FEATURES, N_FEATURES) and np.allclose(cov, cov.T)
    np.linalg.cholesky(cov)                                  # positive definite
    mean = dict(zip(FEATURES, data["mean"]))
    assert mean["tier3"] < mean["tier2"] < 0 < mean["cue_first_sound"]   # sensible directions
    assert data["fit_config"]["n_patients"] >= 100


def test_individual_data_overrides_the_prior_within_a_session(clinical):
    """A patient far from the population (tier 1 uncued p=0.40, prior says ~0.85): after 30 trials the
    posterior has moved most of the way."""
    b = LogisticThompsonBandit(load_logistic_bandit(), clinical, seed=0)
    s = State(last_trials=[], current_tier=1, trial_count=0, session_minutes=0.0)
    i = action_index(1, "none")
    prior_p = b.predict(s)[i]
    rng = np.random.default_rng(3)
    for _ in range(30):
        c = bool(rng.random() < 0.40)
        b.update(s, Action(tier=1, cue="none"),
                 Outcome(correct=c, error_type="correct" if c else "no_response", response_time=3.0))
    post_p = b.predict(s)[i]
    print(f"prior p={prior_p:.2f} -> posterior p={post_p:.2f} after 30 trials (truth 0.40)")
    assert prior_p > 0.75 and post_p < 0.55
