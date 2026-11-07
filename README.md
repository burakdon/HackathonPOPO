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

Implemented and tested (33 tests, `uv run pytest`):
- Agent service (FastAPI) implementing the agreed State/Action interface, with one online-learning
  agent per session and clinician overrides logged for later training.
- Therapist agent: a Beta-Bernoulli Thompson-sampling bandit over the 15 (tier, cue) actions, with
  clinical constraints enforced as action masks.
- End-to-end smoke test: a simple simulated patient plays a full session over HTTP.

In progress or planned:
- Patient app and clinician dashboard (`frontend/`).
- Speech scoring and error classification.
- Patient simulator with population sampling and held-out evaluation patients.
- Contextual bandit and a trained agent (reinforcement learning in simulation).
- Full evaluation against baselines with confidence intervals.
- Expert alignment from clinician overrides.

## Quick start

```bash
uv sync                                         # Python 3.11 env from uv.lock
uv run pytest                                   # test suite
uv run uvicorn service.app:app --port 8000      # agent service
uv run python -m scripts.service_smoke --url http://localhost:8000   # 40-trial session over HTTP
```

## Repository layout

RL and agent code lives at the repository root; the frontend app goes in `frontend/`.

```
core/      shared schema (State/Action/Outcome), typed config loading, action masks
agents/    therapist policies behind one interface (agents/base.py)
service/   FastAPI agent service the frontend calls
sim/       simulated patient population (Step 1b)
envs/ train/ eval/   Gymnasium env, training scripts, evaluation harness (later steps)
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
| Learning from demonstrations (DITTO) | Shaikh et al. (2025). Show, Don't Tell: Aligning Language Models with Demonstrated Feedback. ICLR 2025. | planned (Step 5) |
