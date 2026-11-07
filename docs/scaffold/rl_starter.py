"""
Aphasia Trainer: RL starter kit.

What's in here (one file, no GPU needed):
  1. ACTION SPACE  - tier (1-3) x cue (none ... semantic), matching the README JSON
  2. SimPatient    - a minimal simulated patient (replace with the full simulator
                     later; keep the same methods)
  3. Policies      - FixedPolicy, StaircasePolicy, ThompsonPolicy (the bandit baseline)
  4. run_session   - plays N trials of any policy against any patient, returns the trace
  5. evaluate      - the project's metrics: time in band, learning gain, frustration, cue efficiency
  6. AphasiaEnv    - a Gymnasium env around the same simulator, ready for PPO (stable-baselines3)

Run:  python docs/scaffold/rl_starter.py
Needs: numpy  (gymnasium + stable-baselines3 only for the PPO step)
"""
import json
import math
import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- 1. action space
TIERS = [1, 2, 3]
CUES = ["none", "first_sound", "first_letter", "sentence_frame", "semantic"]
CUE_STRENGTH = {"none": 0, "first_letter": 1, "sentence_frame": 2, "semantic": 2, "first_sound": 3}
BAND = (0.75, 0.90)   # rolling-accuracy band from the project spec
TARGET = 0.85
WINDOW = 8            # rolling window (last 8 trials)


def action_json(tier, cue):
    """The Action contract agreed in the README."""
    return {"tier": int(tier), "cue": cue}


# ---------------------------------------------------------------- 2. simulated patient
@dataclass
class SimPatient:
    """Minimal parametric patient. All parameters are hidden from the agent.

    p(correct) = sigmoid(ability - difficulty(tier) + cue_boost - fatigue)
    Ability grows a little when the patient succeeds on items that are hard for them
    (the 'learning zone'); fatigue grows with trials and with failure streaks.
    """
    name: str = "mild"
    ability: float = 1.2                       # higher = better word retrieval
    cue_response: dict = field(default_factory=lambda: {
        "none": 0.0, "first_letter": 0.4, "sentence_frame": 0.6, "semantic": 0.5, "first_sound": 0.9})
    fatigue_rate: float = 0.01
    learn_rate: float = 0.04
    seed: int = 0

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        self.start_ability = self.ability
        self.fatigue = 0.0
        self.fail_streak = 0

    @staticmethod
    def difficulty(tier):
        return {1: 0.0, 2: 1.2, 3: 2.4}[tier]

    def p_correct(self, tier, cue):
        x = self.ability - self.difficulty(tier) + self.cue_response[cue] - self.fatigue
        return 1 / (1 + math.exp(-x))

    def respond(self, tier, cue):
        p = self.p_correct(tier, cue)
        correct = self.rng.random() < p
        # learning: success on a challenging item (p not too high), with weak cues, teaches most
        if correct and p < 0.95:
            self.ability += self.learn_rate * (1 - p) * (1 - CUE_STRENGTH[cue] / 4)
        self.fail_streak = 0 if correct else self.fail_streak + 1
        self.fatigue += self.fatigue_rate * (1 + self.fail_streak)
        if correct:
            error = "correct"
        else:
            error = self.rng.choice(["sound", "meaning", "no_response"])
        rt = round(self.rng.uniform(1.0, 3.0) + (0 if correct else 1.5), 2)
        return {"correct": correct, "error_type": error, "response_time": rt}


def make_profiles(seed=0):
    return {
        "mild": SimPatient("mild", ability=1.8, fatigue_rate=0.006, seed=seed),
        "moderate": SimPatient("moderate", ability=0.9, fatigue_rate=0.01, seed=seed + 1,
                               cue_response={"none": 0.0, "first_letter": 0.3, "sentence_frame": 0.9,
                                             "semantic": 0.7, "first_sound": 0.5}),
        "severe": SimPatient("severe", ability=-0.2, fatigue_rate=0.015, seed=seed + 2),
    }


# ---------------------------------------------------------------- 3. policies
class FixedPolicy:
    """What worksheets do today: same difficulty, no adaptation."""
    name = "fixed"

    def __init__(self, tier=2, cue="none"):
        self.tier, self.cue = tier, cue

    def act(self, state):
        return action_json(self.tier, self.cue)

    def update(self, state, action, outcome):
        pass


class StaircasePolicy:
    """Classic rule: up after 2 correct in a row, down after 1 error. No cues."""
    name = "staircase"

    def __init__(self):
        self.tier, self.streak = 2, 0

    def act(self, state):
        return action_json(self.tier, "none")

    def update(self, state, action, outcome):
        if outcome["correct"]:
            self.streak += 1
            if self.streak >= 2:
                self.tier, self.streak = min(3, self.tier + 1), 0
        else:
            self.tier, self.streak = max(1, self.tier - 1), 0


class ThompsonPolicy:
    """Bandit baseline. Keeps a Beta posterior of p(correct) for every (tier, cue) arm.

    Each trial it samples p from the allowed arms and picks the one whose sample is
    closest to the 85% target, with a tiny bonus for harder tiers and a cost for stronger
    cues. Like an SLP, it only offers a cue right after an error. Tier moves one step at
    a time. Counts are discounted so it keeps up with fatigue and learning.
    Learns on a live human in a few dozen trials. No GPU.
    """
    name = "bandit"
    PRIOR = {1: (4.0, 1.0), 2: (2.0, 2.0), 3: (1.0, 3.0)}   # weak prior: tier 1 easiest

    def __init__(self, tier_bonus=0.01, cue_cost=0.03, discount=0.95, seed=0):
        self.a = {(t, c): self.PRIOR[t][0] for t in TIERS for c in CUES}
        self.b = {(t, c): self.PRIOR[t][1] for t in TIERS for c in CUES}
        self.tier_bonus, self.cue_cost, self.discount = tier_bonus, cue_cost, discount
        self.rng = np.random.default_rng(seed)
        self.current_tier = 2

    def act(self, state):
        last = (state.get("last_trials") or [{"outcome": True}])[-1]
        cues = CUES if not last["outcome"] else ["none"]
        allowed = [t for t in TIERS if abs(t - self.current_tier) <= 1]
        best, best_score = None, -1e9
        for t in allowed:
            for c in cues:
                p = self.rng.beta(self.a[(t, c)], self.b[(t, c)])
                score = -abs(p - TARGET) + self.tier_bonus * t - self.cue_cost * CUE_STRENGTH[c]
                if score > best_score:
                    best, best_score = (t, c), score
        self.current_tier = best[0]
        return action_json(*best)

    def update(self, state, action, outcome):
        arm = (action["tier"], action["cue"])
        a0, b0 = self.PRIOR[arm[0]]
        self.a[arm] = a0 + self.discount * (self.a[arm] - a0) + (1 if outcome["correct"] else 0)
        self.b[arm] = b0 + self.discount * (self.b[arm] - b0) + (0 if outcome["correct"] else 1)


# ---------------------------------------------------------------- 4. session runner
def make_state(history, tier, trial, minutes):
    """The State contract agreed in the README (last 10 trials + current tier + counters)."""
    return {"last_trials": history[-10:], "current_tier": tier, "trial_count": trial,
            "session_minutes": round(minutes, 1)}


def run_session(policy, patient, n_trials=40):
    history, minutes, tier = [], 0.0, 2
    for i in range(n_trials):
        state = make_state(history, tier, i, minutes)
        action = policy.act(state)
        out = patient.respond(action["tier"], action["cue"])
        policy.update(state, action, out)
        tier = action["tier"]
        minutes += out["response_time"] / 60 + 0.1
        history.append({"word": f"item_{i}", "tier": action["tier"], "cue": action["cue"],
                        "outcome": out["correct"], "error_type": out["error_type"],
                        "response_time": out["response_time"]})
    return {"trace": history, "learning_gain": patient.ability - patient.start_ability}


# ---------------------------------------------------------------- 5. metrics (from the project spec)
def evaluate(result):
    trace = result["trace"]
    acc = [t["outcome"] for t in trace]
    roll = [np.mean(acc[max(0, i - WINDOW + 1): i + 1]) for i in range(len(acc))]
    in_band = np.mean([BAND[0] <= r <= BAND[1] for r in roll[WINDOW:]]) if len(roll) > WINDOW else 0.0
    streak, frustration = 0, 0
    for a in acc:
        streak = 0 if a else streak + 1
        if streak == 3:
            frustration += 1
    correct_cues = [CUE_STRENGTH[t["cue"]] for t in trace if t["outcome"]]
    return {"time_in_band": float(in_band), "learning_gain": float(result["learning_gain"]),
            "frustration_events": frustration,
            "cue_efficiency": float(np.mean(correct_cues)) if correct_cues else 0.0}


def session_reward(result):
    """Episode-level reward for RL training, written out as in the project spec."""
    trace = result["trace"]
    acc = [t["outcome"] for t in trace]
    r, streak = 0.0, 0
    for i, t in enumerate(trace):
        roll = np.mean(acc[max(0, i - WINDOW + 1): i + 1])
        if i >= WINDOW and BAND[0] <= roll <= BAND[1]:
            r += 1.0                                   # + in band
        streak = 0 if t["outcome"] else streak + 1
        if streak == 3:
            r -= 3.0                                   # - three failures in a row
        if i > 0 and abs(t["tier"] - trace[i - 1]["tier"]) > 1:
            r -= 2.0                                   # - tier jump
        if t["outcome"] and CUE_STRENGTH[t["cue"]] >= 2 and i > 0 and trace[i - 1]["outcome"]:
            r -= 0.1                                   # - small: cue probably wasted
    r += 2.0 * (trace[-1]["tier"] - trace[0]["tier"])  # + difficulty climbed
    return r


def compare(n_sessions=200, n_trials=40):
    """The result slide: every policy x every profile, averaged over n_sessions."""
    makers = {"fixed": FixedPolicy, "staircase": StaircasePolicy, "bandit": ThompsonPolicy}
    rows = []
    for prof in ["mild", "moderate", "severe"]:
        for pname, make in makers.items():
            ms = []
            for s in range(n_sessions):
                patient = make_profiles(seed=s * 10)[prof]
                pol = make(seed=s) if pname == "bandit" else make()
                ms.append(evaluate(run_session(pol, patient, n_trials)))
            rows.append({"profile": prof, "policy": pname,
                         **{k: round(float(np.mean([m[k] for m in ms])), 3) for k in ms[0]}})
    return rows


# ---------------------------------------------------------------- 6. Gymnasium env for PPO
try:
    import gymnasium as gym
    from gymnasium import spaces

    class AphasiaEnv(gym.Env):
        """Same simulator as a Gymnasium env, so stable-baselines3 PPO can train on CPU.

        obs:    rolling accuracy, current tier, last 3 outcomes, fail streak, trial progress
        action: index into TIERS x CUES (15 discrete actions)
        reward: per-step version of session_reward
        """
        metadata = {"render_modes": []}
        ACTIONS = [(t, c) for t in TIERS for c in CUES]

        def __init__(self, profile="mild", n_trials=40):
            super().__init__()
            self.profile, self.n_trials = profile, n_trials
            self.action_space = spaces.Discrete(len(self.ACTIONS))
            self.observation_space = spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32)

        def _obs(self):
            acc = self.acc[-WINDOW:] or [0]
            last3 = (self.acc[-3:] + [0, 0, 0])[:3]
            return np.array([np.mean(acc), (self.tier - 1) / 2, *last3,
                             min(self.streak, 5) / 5, self.t / self.n_trials], dtype=np.float32)

        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            s = int(self.np_random.integers(1e6))
            prof = self.profile if self.profile != "random" else ["mild", "moderate", "severe"][s % 3]
            self.patient = make_profiles(seed=s)[prof]
            self.acc, self.tier, self.streak, self.t = [], 2, 0, 0
            return self._obs(), {}

        def step(self, a):
            tier, cue = self.ACTIONS[int(a)]
            r = -2.0 if abs(tier - self.tier) > 1 else 0.0
            out = self.patient.respond(tier, cue)
            self.acc.append(int(out["correct"]))
            self.streak = 0 if out["correct"] else self.streak + 1
            roll = np.mean(self.acc[-WINDOW:])
            if len(self.acc) >= WINDOW and BAND[0] <= roll <= BAND[1]:
                r += 1.0
            if self.streak == 3:
                r -= 3.0
            r += 0.5 * (tier - self.tier) if BAND[0] <= roll <= BAND[1] else 0.0
            r -= 0.02 * CUE_STRENGTH[cue]
            self.tier, self.t = tier, self.t + 1
            return self._obs(), r, self.t >= self.n_trials, False, {}
except ImportError:
    AphasiaEnv = None


if __name__ == "__main__":
    print("Example action JSON:", json.dumps(ThompsonPolicy().act({})))
    print("\nComparison on simulated patients (200 sessions x 40 trials each):\n")
    hdr = f"{'profile':9} {'policy':10} {'in band':>8} {'gain':>7} {'frustr.':>8} {'cue str.':>8}"
    print(hdr + "\n" + "-" * len(hdr))
    for r in compare():
        print(f"{r['profile']:9} {r['policy']:10} {r['time_in_band']:8.2f} {r['learning_gain']:7.3f} "
              f"{r['frustration_events']:8.2f} {r['cue_efficiency']:8.2f}")
    print("\nNext: train PPO on AphasiaEnv (pip install gymnasium stable-baselines3):")
    print("  from stable_baselines3 import PPO; from docs.scaffold.rl_starter import AphasiaEnv")
    print("  model = PPO('MlpPolicy', AphasiaEnv('random'), verbose=1).learn(100_000)")
