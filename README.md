# Aphasia Trainer

A home practice trainer for people with aphasia, the loss of language that follows many strokes. The
patient names pictures; an AI therapist chooses the difficulty and the hint (cue) for every trial so
the patient stays near 80 to 85 percent correct. The therapist is trained against a population of
simulated patients and runs live on a real person, and a speech-language pathologist (SLP) can
override any of its choices.

> **Practice tool used under an SLP.** It does not diagnose, replace therapy, or make treatment
> decisions. No real patient data is used anywhere. We do not claim it improves recovery in real
> patients.

## The system

| component | what it does |
|---|---|
| Patient app | Browser app: shows a picture, takes a spoken or tapped answer, shows the cue. Large type, one thing on screen. |
| Speech scorer | Transcribes the answer and labels it: correct, sound error, meaning error, or no response. |
| Therapist agent | Chooses the next difficulty tier and cue from the session so far (`agents/`, served by `service/`). |
| Patient simulator | A population of simulated patients used to train and evaluate the therapist (`sim/`). |
| Clinician dashboard | Session trace (difficulty, accuracy, error types, the agent's choices) with an override on every choice. |
| Expert-feedback loop | Clinician overrides are stored as demonstrations so the therapist can learn a clinician's preferences. |

Live loop: the patient names a picture, the scorer labels the answer, the therapist picks the next
picture's tier and cue, and the app shows it. Two learning signals feed the therapist: practice on
simulated patients, and demonstrations from clinician overrides.

## Status

Implemented and tested (117 tests, `uv run pytest`):
- Agent service (FastAPI) implementing the agreed State/Action interface, with one online-learning
  agent per session, clinician overrides logged for later training, and CORS for a frontend on the
  local network.
- Patient simulator: population sampling with a held-out test split, three evaluation cohorts,
  item difficulty from word frequency and length, patient-specific cue responsiveness, error types,
  fatigue with recovery, and learning with a configurable peak. The item bank is a placeholder.
- Therapist agents: a contextual Bayesian logistic Thompson-sampling bandit with a prior fitted on
  simulated training patients (the live agent), a Beta-Bernoulli Thompson bandit, and fixed-difficulty
  and staircase baselines.
- Trained agent: a Gymnasium environment with clinical action masks and a decomposed proxy reward,
  and MaskablePPO trained on simulated patients (3 seeds, learning curves on held-out patients).
- Preliminary evaluation: five policies on three cohorts, 50 held-out patients each, with patient
  dropout on and off (see below).
- Language-model therapist: prompt format, offline reward and dataset, and a 10-step Tinker training
  smoke test (not yet run against the API).

In progress or planned:
- Patient app and clinician dashboard (`frontend/`).
- Speech scoring and error classification.
- Language-model therapist trained with group-relative policy optimisation (GRPO), and a prompted
  frontier-model baseline.
- Full evaluation: privileged oracle, 200+ patients per cohort, several training seeds, bootstrap
  confidence intervals, and reward and learning-peak ablations.
- Expert alignment from clinician overrides.
- A clinically curated item bank and clinical validation of the simulator's parameters.

## Quick start

```bash
uv sync                                         # Python 3.11 env from uv.lock
uv run pytest                                   # test suite
uv run uvicorn service.app:app --host 0.0.0.0 --port 8000   # agent service, reachable on the LAN
uv run python -m scripts.service_smoke --url http://localhost:8000   # 40-trial session over HTTP
uv run python -m train.fit_bandit_prior         # refit the bandit's empirical-Bayes prior (~2 s)
uv run python -m train.train_ppo                # MaskablePPO, 3 seeds (~3 min each on CPU) -> models/
uv run python -m eval.plots                     # learning curves -> results/learning_curves.png
uv run python -m scripts.sanity_table --dropout on --out results/sanity_dropout_on.md
uv run python -m scripts.sanity_table --dropout off --out results/sanity_dropout_off.md
uv sync --extra grpo && uv run python -m train.grpo_smoke   # needs TINKER_API_KEY in .env
```

**Frontend on another laptop:** run the service with `--host 0.0.0.0`, find this machine's LAN IP
(`ipconfig getifaddr en0` on macOS) and call `http://<that-ip>:8000`. CORS allows the Vite dev server
(port 5173) on localhost and on any private-network address (`cors_origins` / `cors_origin_regex` in
[`configs/service.yaml`](configs/service.yaml)). The macOS firewall may ask to allow incoming
connections for Python the first time.

## Repository layout

RL and agent code lives at the repository root; the frontend app goes in `frontend/`.

```
core/      shared schema (State/Action/Outcome), typed config loading, action masks
agents/    therapist policies behind one interface (agents/base.py)
service/   FastAPI agent service the frontend calls
sim/       simulated patient population
envs/      Gymnasium environment with action masks
train/     PPO training, bandit-prior fitting, language-model (GRPO) training
eval/      metrics and figures
models/    trained PPO policies (small, committed so results reproduce without retraining)
configs/   all tunable values (YAML)
scripts/   smoke tests and utilities
tests/     pytest suite
results/   committed CSV/PNG outputs for this README
docs/      starter scaffold kept for reference (docs/scaffold/rl_starter.py)
frontend/  patient app and clinician dashboard
```

## Branches

- `main`: stable and demo-ready only. Updated from `dev` at checkpoints, when the demo works end to end.
- `dev`: integration branch. Small fixes (CORS, typos, README) may be committed here directly.
- `feature/rl-agent`: all RL work (simulator, bandits, training, evaluation), branched from `dev`.

Merges happen only through pull requests on GitHub (`feature/rl-agent` → `dev`, and `dev` → `main`
at checkpoints), opened once the tests pass and a piece works. Nobody merges locally.

## Interfaces

The contract between the frontend, the scorer and the agent. Defined once in
[`core/schema.py`](core/schema.py) (pydantic); the service validates every request against it and
returns HTTP 422 with a precise error on any violation.

**Changes from the original contract (both additive and backward-compatible):**
1. `State` has an optional `session_id`. The service keeps one online-learning agent per session;
   requests without it share a single `default` session. **Please always send it** (one id per
   patient session, any string up to 128 chars).
2. `POST /update` takes an explicit `outcome` object (shape below).

### State
```json
{"session_id": "patient-42-2026-11-07",
 "last_trials": [{"word": "cat", "tier": 2, "cue": "first_sound", "outcome": false,
                  "error_type": "sound", "response_time": 3.1}],
 "current_tier": 2, "trial_count": 14, "session_minutes": 6.5}
```
| field | type | rule |
|---|---|---|
| `session_id` | string, optional | 1 to 128 chars |
| `last_trials` | list | most recent last; up to 10 (longer lists are truncated to the last 10) |
| `last_trials[].tier` | int | 1..3 |
| `last_trials[].cue` | string | `none \| first_sound \| first_letter \| sentence_frame \| semantic` |
| `last_trials[].outcome` | bool | must be `true` iff `error_type == "correct"` |
| `last_trials[].error_type` | string | `correct \| sound \| meaning \| no_response` |
| `last_trials[].response_time` | float | seconds, >= 0 |
| `current_tier` | int | 1..3, the tier of the most recent trial (the agent plans from this) |
| `trial_count` | int | >= 0 |
| `session_minutes` | float | >= 0 |

Unknown extra fields are ignored, so adding e.g. a timestamp will not break anything.

### Action
```json
{"tier": 2, "cue": "first_sound", "rationale": "short text for the dashboard"}
```
`tier` in 1..3, `cue` as above. `rationale` is a human-readable explanation (may be null).
The agent never returns a tier more than one step from `current_tier`, and never a cue outside the
clinician-approved set in [`configs/clinical.yaml`](configs/clinical.yaml).

### Outcome
```json
{"correct": false, "error_type": "sound", "response_time": 3.1}
```
`correct` must be `true` iff `error_type == "correct"`.

### Endpoints
| method | path | body | returns |
|---|---|---|---|
| GET | `/health` | none | `{"status": "ok", "agent": "...", "active_sessions": n}` |
| POST | `/act` | `State` | `Action` |
| POST | `/update` | `{"state": State, "action": Action, "outcome": Outcome}` | `{"status": "ok", "session_id": "..."}` |
| POST | `/override` | `{"state": State, "agent_action": Action, "clinician_action": Action, "note": "optional"}` | `{"status": "stored", "session_id": "...", "n_overrides": n}` |

Session loop for the frontend:
1. `POST /act` with the current `State` and show the returned tier and cue.
2. If the clinician overrides, `POST /override` with both actions and show the clinician's action.
3. After the patient answers, `POST /update` with the same `State`, the action **actually shown**
   (the clinician's if overridden), and the `Outcome`.
4. Append the trial to `last_trials` and repeat.

Overrides are stored as JSON lines in `runs/overrides.jsonl` (the input to the DITTO step).
Interactive API docs: `http://localhost:8000/docs` while the service runs.

## Problem formulation

A **POMDP**. The patient's true state (retrieval ability per item, fatigue, responsiveness to each
cue type, learning rate) is hidden; the agent only observes trial outcomes through the State JSON.
An **episode** is one session of 40 trials; a **step** is one trial. The **action** is (tier 1..3) x
(cue: 5 types) = 15 discrete actions. Clinical constraints are **action masks**, not penalties: tier
moves at most one step per trial, and cues outside the clinician-approved set are never offered.

## Simulated patients (`sim/`)

`p(correct) = sigmoid(ability + item_learning[i] - difficulty[i] + cue_gain[cue] - fatigue)`, all on
the log-odds scale. Every parameter is in [`configs/sim.yaml`](configs/sim.yaml) with a comment that
either cites the paper behind the *direction* of the effect or says `ASSUMPTION`.

| component | model | grounding |
|---|---|---|
| item difficulty | `beta_f * (-z(frequency)) + beta_l * z(length)` | frequency and length effects on naming (Nickels & Howard 1995) |
| cues | `phonemic_weight[cue] * phon_resp + semantic_weight[cue] * sem_resp`, per-patient responsiveness | phonemic > semantic on average, with phonemic vs semantic responders (Meteyard & Bose 2018) |
| error types | after a failure, P(sound, meaning, no response) proportional to (phon_impair, sem_impair, omission_bias) | simplified from the two-step model (Dell et al. 1997; Schwartz et al. 2006) |
| fatigue | grows per trial and faster during failure streaks, recovers a fraction each trial, capped | ASSUMPTION |
| learning | on a success, item-specific gain with E[gain] = p^a (1-p)^b, peaking at a configurable `peak`; weaker cues teach more; 5% spills over to general ability | item-specific gains with limited generalisation (Nickels 2002); the peak location is an ASSUMPTION; a learning-peak ablation is planned |
| response time | log-normal, slower for hard items and when fatigued | ASSUMPTION |
| dropout | after 3 failures in a row the patient may quit: 5% per trial, +5% per further failure | ASSUMPTION (conservative); results reported with dropout on and off |

**Population, not fixed profiles.** Training patients are drawn from broad parameter distributions
(domain randomisation). Three named presets (mild anomic, moderate Broca's, severe) are narrow
cohorts used only for evaluation. Patient *k* of a split comes from `SeedSequence([seed, split, cohort, k])`,
so the **train and test populations are disjoint** (tested), and every policy faces exactly the same
patients with the same random outcome stream (common random numbers, so comparisons are paired).

**Item bank.** Loaded from a CSV with columns `word,image_file,frequency,length,tier` (schema in
[`sim/items.py`](sim/items.py)). The committed bank [`data/items_placeholder.csv`](data/items_placeholder.csv)
is a **PLACEHOLDER**: 60 picturable nouns, frequencies from the `wordfreq` package (Zipf scale),
length in letters, tiers = difficulty terciles. It is to be replaced by a clinically curated bank. Within a tier,
items are shown unseen-first, then least recently seen; the app should use the same rule.

## Agents (`agents/`)

All agents implement one interface (`act(state, mask) -> Action`, `update(state, action, outcome)`)
and decide from the State JSON only.

- **Fixed difficulty** (worksheets today) and **2-up/1-down staircase** (Levitt 1971).
- **Beta-Bernoulli Thompson bandit**: 15 independent arms, ported from the scaffold and bug-fixed.
- **Contextual Bayesian logistic Thompson bandit** (the live agent): `p(correct) = sigmoid(w . phi)`
  with 17 features (tier, cue, recent accuracy, fail streak, progress, response time, and last error
  type x cue family; the simulator does not yet model that interaction, see below). Posterior updated online with the closed-form Laplace/extended-Kalman step,
  with random-walk process noise so it tracks fatigue and learning. Weights are shared across
  actions, so one observation informs many actions.
- **Empirical-Bayes prior** for the contextual bandit, fitted on 300 simulated *training* patients
  (`train/fit_bandit_prior.py`): the per-patient weights' mean and shrunk covariance. This is the
  bandit's sim-to-real pretraining. It is inflated so a real person's data still dominates: in a test,
  a patient far from the population moves the prediction from 0.85 to 0.44 in 30 trials (truth 0.40).
- **Shared selection rule** (`agents/selection.py`): among actions whose predicted p is within 0.06
  of the 0.85 target, take the hardest tier, then the weakest cue; otherwise the closest to target.
  This fixes the tier-1 coasting of the first version of the bandit (37 of 40 trials on tier 1).

## Trained agent: MaskablePPO (`envs/`, `train/train_ppo.py`)

**Environment.** One episode is one 40-trial session with a new simulated patient drawn from the
broad training population. The observation is built from the State JSON only (the last 8 trials as
one-hot tier, cue and error type plus response time, and the current tier, rolling accuracy, fail
streak and session progress), so the trained policy runs unchanged behind the service. The history
stack gives the policy memory in this POMDP. The action masks are the same clinical masks the
bandits use, including "cue only after an error". (sb3-contrib has no recurrent *and* masked PPO,
so we use MaskablePPO with a history stack rather than RecurrentPPO, which would turn masks into
penalties.)

**Proxy reward** ([`configs/reward.yaml`](configs/reward.yaml)), every term observable on a real
patient: +1 per trial with rolling accuracy in the band; -1 per trial while the failure streak is 3
or more; -0.1 when a correct uncued answer follows a correct answer with a heavy cue (cue probably
wasted); and, once at the end, +2 per tier climbed, paid only if the session spent at least half its
trials in band. Making the climb bonus terminal and conditional removes an exploit in the original
scaffold (per-step reward for climbing while in band, no cost for dropping out of band, so
oscillating tiers farmed reward). Tests check that oscillating tiers earns nothing per step and,
in the environment, that alternating tiers 1 and 2 scores less than holding tier 2 (mean return
14.2 vs 18.4 on validation patients). The true objective, learning gain, is never part of the reward.

**Training.** 3 seeds x 1.5M steps, CPU, about 3 minutes per seed. Learning curves are measured on
64 fixed validation patients that are disjoint from both the training and the test patients.
Validation time in band rises from 0.41 (untrained) to about 0.73 within 100k steps and then
plateaus; true control error improves only from about 0.17 to 0.14.

![PPO learning curves](results/learning_curves.png)

## Preliminary results

50 held-out test patients per cohort, 40 trials, dropout **on**, mean ± 95% CI over patients (normal
approximation); PPO metrics are averaged over its 3 seeds for each patient. Every policy faces the
same patients with the same random outcome stream, so differences are due to the policy. Full
tables: [dropout on](results/sanity_dropout_on.md), [dropout off](results/sanity_dropout_off.md).
"learn eff." is the mean expected learning per trial implied by the simulator's learning rule at the
true p (1.0 = every trial at the learning peak with no cue); it explains the learning-gain column.

| cohort | policy | in band | acc | mean tier | gain x100 | learn eff. | \|p-0.85\| | frustration | quit |
|---|---|---|---|---|---|---|---|---|---|
| mild anomic | fixed (tier 2) | 0.50 ± 0.06 | 0.90 | 2.00 | **3.33 ± 0.29** | **0.79** | **0.066** | 0.02 | 0.00 |
| mild anomic | staircase | 0.80 ± 0.03 | 0.79 | 2.29 | 2.98 ± 0.23 | 0.64 | 0.143 | 0.10 | 0.02 |
| mild anomic | Beta bandit | 0.41 ± 0.07 | 0.94 | 1.38 | 2.39 ± 0.24 | 0.56 | 0.095 | 0.00 | 0.00 |
| mild anomic | contextual bandit | 0.52 ± 0.06 | 0.91 | 1.77 | 2.89 ± 0.27 | 0.67 | 0.085 | 0.00 | 0.00 |
| mild anomic | PPO (3 seeds) | **0.89 ± 0.02** | 0.85 | 1.93 | 2.55 ± 0.21 | 0.56 | 0.126 | 0.01 | 0.00 |
| moderate Broca's | fixed (tier 2) | 0.53 ± 0.06 | 0.74 | 2.00 | 4.63 ± 0.41 | 0.74 | 0.109 | 0.62 | 0.04 |
| moderate Broca's | staircase | 0.54 ± 0.04 | 0.70 | 1.87 | 3.95 ± 0.29 | 0.65 | 0.169 | 0.28 | 0.04 |
| moderate Broca's | Beta bandit | 0.61 ± 0.05 | 0.84 | 1.28 | 4.75 ± 0.33 | **0.81** | **0.068** | 0.14 | 0.00 |
| moderate Broca's | contextual bandit | 0.63 ± 0.05 | 0.83 | 1.42 | **4.80 ± 0.32** | **0.81** | 0.069 | 0.24 | 0.00 |
| moderate Broca's | PPO (3 seeds) | **0.86 ± 0.03** | 0.79 | 1.57 | 4.16 ± 0.30 | 0.68 | 0.116 | 0.05 | 0.00 |
| severe | fixed (tier 2) | 0.02 ± 0.01 | 0.28 | 2.00 | 0.10 ± 0.05 | 0.02 | 0.521 | 2.08 | 0.82 |
| severe | staircase | 0.08 ± 0.03 | 0.49 | 1.25 | 1.18 ± 0.26 | 0.22 | 0.350 | 1.76 | 0.36 |
| severe | Beta bandit | 0.20 ± 0.05 | 0.53 | 1.31 | 1.49 ± 0.31 | 0.25 | 0.308 | 1.44 | 0.26 |
| severe | contextual bandit | 0.25 ± 0.06 | 0.57 | 1.19 | 1.86 ± 0.35 | 0.29 | 0.267 | 1.20 | 0.18 |
| severe | PPO (3 seeds) | **0.34 ± 0.06** | 0.60 | 1.10 | **2.10 ± 0.33** | **0.33** | **0.244** | **0.81** | **0.13** |

What this shows, honestly:
- **PPO holds the band best in every cohort** (0.89, 0.86, 0.34) and is the best policy overall on
  severe patients (highest learning gain, lowest control error, fewest frustration events and
  quits).
- **On mild and moderate patients PPO partly games the proxy.** Its true control error (0.126,
  0.116) is worse than the contextual bandit's (0.085, 0.069), and so is its learning gain. It has
  learned to steer the 8-trial rolling accuracy (an easier item after an error, a harder one after a
  success) rather than to keep the true p near 0.85. The staircase shows the same pattern (0.80 in
  band on mild with the worst control error). This is why time in band is never reported alone.
- **The contextual bandit is the most accurate controller of true difficulty** on mild and
  moderate patients and has the highest learning gain among adaptive policies there; it is the
  current live agent because it also learns online on a real person.
- **Severe patients remain hard for every policy.** One cause is the clinical rule "cue only after
  an error": their best action (tier 1 + first sound, p~0.84) cannot be repeated after a success.
  With the rule off, the contextual bandit's time in band rises on mild and severe patients but
  falls on moderate ones (measured with the hand-set prior). Whether to keep the rule is to be
  validated clinically (`cue_only_after_error`).
- The Beta bandit still spends 63% of mild sessions on tier 1 despite the progression rule: its
  discounting pulls unvisited arms back to the tier-1 prior mean 0.80, which sits inside the
  tolerance window. Shared weights in the contextual model avoid this.

**Dropout on vs off.** With dropout off, the mild and moderate rows are unchanged within their
confidence intervals (almost nobody has 3+ failures in a row there). The effect is on severe
patients and on policies that let failures accumulate: fixed tier 2 quits in 82% of severe sessions
(20.5 of 40 trials on average) and its learning gain falls from 0.12 to 0.10, the staircase quits in
36%, while PPO (13%) and the contextual bandit (18%) keep most sessions going. Dropout is therefore
the mechanism by which frustration costs practice volume, and it widens the gap between adaptive
and fixed difficulty only where the fixed tier is wrong for the patient.

### Why fixed difficulty can win learning gain in the simulator

Fixed tier 2 has the highest learning gain on mild patients (3.33) and is close to the best on
moderate ones. We did not tune the simulator to reverse this. The mechanism is visible in the
"learn eff." column: learning gain tracks how much of the session is spent near the simulator's
learning peak (p = 0.85) **without cues** (successes with cues teach less, by assumption).

- For mild patients, tier 2 happens to give p of about 0.90, right next to the peak, every trial,
  with no cue (learn eff. 0.79). Adaptive policies spend part of the session exploring, start on
  tier 1 where p is about 0.96 and little is learned, and use cues after errors, all of which lower
  learning per trial.
- It is the *matching* that matters, not "fixed": on mild patients fixed tier 1 gets 1.73 and fixed
  tier 3 gets 2.51; on moderate patients fixed tier 1 gets 4.92 (the best there) while fixed tier 3
  gets 0.36 with 82% of sessions abandoned. Picking the right fixed tier for each patient requires
  exactly the knowledge an adaptive policy acquires; worksheets do not make that choice per patient.

What this **does** mean: in this simulator, with a single 40-trial session, a policy that sits at
the learning peak from the first trial is hard to beat, and exploration has a real cost.
What it **does not** mean: that fixed difficulty is a good strategy. The result depends on
assumptions we cannot validate yet (the location of the learning peak, the cue discount, item-specific
learning, a single session with no boredom or disengagement effects for items that are too easy) and
it reverses as soon as the fixed tier does not match the patient. A learning-peak ablation and
multi-session evaluation are planned.

**Error-type x cue features.** The contextual bandit has six features crossing the last error type
(sound, meaning, no response) with the cue family (phonemic, semantic). **The simulator does not
model that interaction yet**: a patient's response to a cue does not depend on the error they just
made. Accordingly these weights fit to about 0 in the empirical-Bayes prior. They are kept because
real patients may show the dependency (for example, phonemic cues after sound errors), and they cost
little.

## Known limitations
- With an 8-trial window, rolling accuracy only takes values k/8, so the [0.75, 0.90] band means
  exactly 6/8 or 7/8. Even a perfect controller holding p = 0.85 is in band only ~62% of the time
  (Binomial(8, 0.85)). This is why we also report true control error, and will report % of a privileged oracle.
- The simulator is a parametric model; magnitudes are assumptions pending clinical review. The
  bandit's fitted prior inherits these assumptions.
- One action per trial applies the cue to the *next* item; real cueing happens within an item after
  a failed attempt. Kept for the agreed interface.

## Reproducibility
- Every random source takes a seed. Service sessions derive their seed from
  `(configs/bandit.yaml seed, session_id)` via crc32, so replaying the same requests reproduces the
  same actions (tested in `tests/test_service.py::test_full_session_replays_identically`).
- All tunable values live in `configs/*.yaml` and are validated at load time (`core/config.py`).

## Citations
Every paper cited in code or configs is listed here so it can be checked. Papers are cited only for
the **direction** of an effect; specific numeric values are marked `ASSUMPTION` in the configs.

| cited for | reference | where |
|---|---|---|
| 85% target accuracy (theoretical, not shown for aphasia) | Wilson, Shenhav, Straccia & Cohen (2019). The Eighty Five Percent Rule for optimal learning. *Nature Communications* 10, 4646. | `configs/clinical.yaml` |
| Phonemic cues have larger immediate effects than semantic cues | Meteyard & Bose (2018). What does a cue do? Comparing phonological and semantic cues for picture naming in aphasia. *JSLHR* 61(3), 658–674. | `configs/clinical.yaml` |
| Discounted Thompson sampling for non-stationary bandits | Raj & Kalyani (2017). Taming non-stationary bandits: A Bayesian approach. arXiv:1707.09727. | `configs/bandit.yaml` |
| Word frequency and length predict naming accuracy in aphasia | Nickels & Howard (1995). Aphasic naming: What matters? *Neuropsychologia* 33(10), 1281–1303. | `configs/sim.yaml` (items) |
| Naming therapy gains are largely item-specific, limited generalisation | Nickels (2002). Therapy for naming disorders: Revisiting, revising, and reviewing. *Aphasiology* 16(10–11), 935–979. | `configs/sim.yaml` (learning) |
| Error types reflect semantic vs phonological impairment (two-step model) | Dell, Schwartz, Martin, Saffran & Gagnon (1997). Lexical access in aphasic and nonaphasic speakers. *Psychological Review* 104(4), 801–838. | `configs/sim.yaml` (errors) |
| Case-series evidence for the two-step model in picture naming | Schwartz, Dell, Martin, Gahl & Sobel (2006). A case-series test of the interactive two-step model of lexical access. *Journal of Memory and Language* 54(2), 228–264. | `configs/sim.yaml` (errors) |
| 2-up/1-down staircase converges near 70.7% correct | Levitt (1971). Transformed up-down methods in psychoacoustics. *JASA* 49(2B), 467–477. | `agents/baselines.py` |
| Thompson sampling with a Laplace-approximated logistic model | Chapelle & Li (2011). An empirical evaluation of Thompson sampling. *NeurIPS* 24. | `agents/logistic_bandit.py` |
| Word frequencies for the placeholder item bank | Speer (2022). wordfreq (software), v3. https://github.com/rspeer/wordfreq | `scripts/build_placeholder_items.py` |
| Proximal policy optimisation | Schulman, Wolski, Dhariwal, Radford & Klimov (2017). Proximal Policy Optimization Algorithms. arXiv:1707.06347. | `train/train_ppo.py` |
| Invalid-action masking in policy-gradient methods | Huang & Ontañón (2022). A Closer Look at Invalid Action Masking in Policy Gradient Algorithms. *FLAIRS-35*. | `envs/aphasia_env.py` |
| PPO implementation | Raffin et al. (2021). Stable-Baselines3: Reliable Reinforcement Learning Implementations. *JMLR* 22(268). | `train/train_ppo.py` |
| Group-relative policy optimisation (GRPO) | Shao et al. (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models. arXiv:2402.03300. | `train/grpo_smoke.py` |
| Learning from demonstrations (DITTO) | Shaikh et al. (2025). Show, Don't Tell: Aligning Language Models with Demonstrated Feedback. ICLR 2025. | planned (expert alignment) |
