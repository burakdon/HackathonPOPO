# Person 2 hand-off report (2026-11-08)

Nothing was committed or pushed. Full suite: `uv run --extra llm --extra speech pytest -q` passes (1 pre-existing skip).
`uvx ruff check` (rules E, F, W; line length ignored) is clean on every file I created or changed. The local ruff
setup applies a much stricter default rule set that also flags existing code; I did not clean that up.

## Per phase

| phase | status |
|---|---|
| 1 Corrections and examples | done |
| 2 Note -> rule -> matches -> examples | done, real-key check **not run: no ANTHROPIC_API_KEY** |
| 3 Training runs, preview, activation | done with fakes; real training run **deferred** |
| 4 Hint audio | done |
| 5 Scripted demo history | done |
| 6 Scorer report | done |
| 7 Small fixes | done |

### Phase 1 (tests in `tests/test_steering.py`)
- [pass] correct a decision -> 201; review shows it in `correction`; examples list has it; `pending_examples == 1`.
- [pass] preferred not an option -> 422 (message as in the sample); unknown decision -> 404; correcting twice -> one example (`x-0001`), updated; a used example -> 409.
- [pass] examples, notes and the counters survive an app restart.

### Phase 2
- [pass] fake drafter: note -> draft; confirm; matches on the demo situation (d02: `allowed: true`, rule says first_sound, model chose sentence_frame; the last-try decision d03: `allowed: false`, `rule_says: null`); tick -> examples; ticking a non-allowed decision -> 422.
- [pass] no key -> `rule: null` with "Rule drafting is not configured (no ANTHROPIC_API_KEY); fill in the rule by hand."; `understood=false` -> `rule: null` with its problem; matches on an unconfirmed note -> 409.
- [pass] `RuleDraft` -> `Rule` conversion incl. relative level "easier"; a fake Claude client checks refusal, errors, model override.
- [NOT RUN] manual check with a real key: no `ANTHROPIC_API_KEY` is set. No output to paste. The prompt and the SDK call (`client.messages.parse(..., output_format=RuleDraft)`) have only been exercised against a fake client.
- Decisions I took: a Claude draft with no conditions at all is rejected (problem asks for a hand-written rule); `missed_pictures_in_a_row` is computed from the session history (see contract notes); one example per decision (re-choosing replaces a pending one, a used one is 409).

### Phase 3 (tests in `tests/test_training.py`)
- [pass] fake trainer: start 202; second start 409; no pending 422; progress mapping; completed -> examples `used in <run_id>`, preview with totals; failed run and failed preview; restart persistence; background preview runs once.
- [pass] use-prepared: equal example IDs -> `prepared: true`; different ones, an unfinished or unknown prepared run, or an already finished live run -> 409/404.
- [pass] models list includes the run; activating changes `GET /patients/{pid}` and the next session's `model_version`; 422 for unknown or incomplete versions.
- [pass] real pieces without models: rows written in the demonstration format (same keys, `scope: global`, no `patient_id`), manifest reading of `SubprocessTrainer`, preview situation generator (situations satisfy the rule and the app's option rule), preview counting with scripted models.
- [DEFERRED] manual real DITTO run for `demo-ana` (about 5-6 minutes), to run once Person 3's training update lands:
  1. In the app: write the demo note, confirm the rule, tick the matches (or use corrections), then `POST /patients/demo-ana/training-runs`.
  2. Poll `GET /training-runs/demo-ana-r1`; read `runs/ditto/demo-ana-r1/manifest.json` and `runs/cueagain/training/demo-ana-r1.log` on failure.
  The real previewer (`service/preview.py`) has therefore never run against real adapters; its first real preview numbers are still to be measured.
- Not tested for real: that three or so examples among the 20 seed rows change the model's choice in the demo situation. The full chain "activate, new session gives first_sound" is untested with a real adapter.
- Decisions I took: training starts from the base model (not "from the shared adapter") and uses all the patient's examples; `use-prepared` also requires the same patient and stops the live run's training process; `ditto:<live run>` is served by the prepared run's adapter.

### Phase 4
- [pass] 120 files in `frontend/hints/`; `GET /pictures` returns 96 hint and 24 whole-word non-null `audio_url`s, and a clip is served (`audio/mp4a-latm`).
- [pass] re-running skips existing files (a hand-written file survived); `--force` overwrites. `frontend/hints/README.md` written. First-sound clips are poor text-to-speech and should be re-recorded. The 120 files are untracked (not gitignored); your call whether to commit them.

### Phase 5 (`scripts/seed_demo_sessions.py`, run on the default data dir)
- [pass] three sessions for `demo-ana`, all `simulated: true`, played through real Whisper, the real scorer and the patient's real model (`ditto:demoA-qwen3-1.7b-r1`, 0 fallbacks); transcripts are what Whisper heard (clips are pre-checked and a different candidate word is used if a clip is misheard, but a transcript is never edited).
- [pass] the demo rule has 5 tickable matches (16 matching decisions in all; the rest are last-try decisions where no hint is offered). All 5 recordings play back.
- Caveats: the model starts with a meaning hint only on some stuck pictures, and runs differ (3 to 5 tickable between runs, from timing in the context); I scripted most pictures as "stuck", so the sessions are failure-heavy (most pictures unsuccessful). Timestamps are real (not backdated). `--reset` removes only these sessions (list in `runs/cueagain/seed_demo_sessions.json`).
- The live server on port 8000 holds its data in memory and does not see the new sessions or any new files until restarted. It could also overwrite `patients/demo-ana.json` from its stale copy if it changes the plan or model.

### Phase 6 (`scripts/scorer_report.py`, `results/scorer_synthetic.md` and `.csv`)
- [pass] actual run: 808 clips, 4 voices (Samantha, Daniel, Karen, Moira), Whisper `base.en`; the CSV lists every clip.
- Headline: 10/288 (3.5%) of clips that should be incorrect came back "correct", all first-sound near-misses that Whisper heard as the real word ("panana" -> "Banana", "ombrella" -> "Umbrella", "bog" -> "dog"). No wrong word and no "I don't know" was accepted.
- Also: 93/328 (28.4%) of clips that should be correct did not score correct, mostly Whisper mishearing "cat" (and "puppy" -> "Happy"). Error type was right for 226/250 incorrect answers. Optimistic: synthetic speech (caveat in the report).
- Threshold proposals: none applied. The data points to the recognizer rather than the thresholds (the false rejects are mishearings, not low-confidence rejections); lowering `MIN_WORD_CONFIDENCE` below 0.1 would only convert a few "cats"/"heads"-type unclear results into guesses, so I do not propose it.

### Phase 7
- [done] `restore()` in `service/practice.py` skips JSON files that are not session snapshots. `.env.example` has `ANTHROPIC_API_KEY=` and a commented `CUEAGAIN_RULE_MODEL=`.

## Contract changes (docs/API_CONTRACT.md, core/contract.py, scripts/build_contract_examples.py updated together; samples regenerated)
- Patient adapters are `ditto:<run_id>` (not `SHARED+run`); sample `GET_models` / `POST_active_model` updated.
- `POST /patients/{pid}/active-model` documented: 422 rules, applies to later sessions, returns `Patient`.
- Additive helpers in `core/contract.py`: `situation()` (moved from the generator), `missed_before()`, `matches_rule()`. No model fields changed.
- Preview `label` on `follows_your_rules_new_situations` says the situations are generated.
- Implementation notes added under §8 (listed in the doc).

## New dependency
- `anthropic>=2.6.0` (main dependencies; 2.6.0 locked, plus `docstring-parser`, `jiter`, `sniffio`). `pyproject.toml`'s `cmudict` entry in the `speech` extra was already uncommitted before my work.

## Files
Created: `service/errors.py`, `service/steering.py`, `service/rule_drafter.py`, `service/training.py`, `service/preview.py`, `scripts/make_hint_audio.py`, `scripts/scorer_report.py`, `scripts/seed_demo_sessions.py`, `tests/test_steering.py`, `tests/test_training.py`, `frontend/hints/` (README + 120 clips), `results/scorer_synthetic.md`, `results/scorer_synthetic.csv`, `docs/HANDOFF_PERSON2_REPORT.md`.
Changed: `service/engine.py` (steering and training hooks, `decision_options`, pending count, corrections in review, errors moved), `service/patient_api.py`, `service/app.py` (new `rule_drafter`, `cueagain_trainer`, `cueagain_previewer` arguments), `service/practice.py`, `core/contract.py`, `core/config.py` (`load_dotenv` moved here), `agents/llm_prompted.py`, `scripts/build_contract_examples.py`, `docs/API_CONTRACT.md`, `docs/OPEN_DECISIONS.md`, `.env.example`, `pyproject.toml`, `uv.lock`, `docs/contract_examples/*` (regenerated).
Data (gitignored): `runs/cueagain/` now holds three seeded sessions and `seed_demo_sessions.json`; `runs/scorer_clips/`.

## Known gaps
- No real-key rule drafting, no real training run, no real preview numbers (see above).
- The product UI (`product/`) was not touched; it still needs wiring to the steering and training endpoints.
- A training process cancelled by `use-prepared` leaves its manifest marked "running".

## Update 2026-11-08: per-patient training now starts from the shared adapter

- `SubprocessTrainer.start` (service/training.py) now runs `python -m training.patient_ditto --run-id <id>
  --examples <file>` instead of `python -m training.ditto --run-id <id> --extra-demos <file>`. Same run ids, same
  examples file (`example_rows`), same `runs/ditto/<id>/manifest.json` format, so `status()`, the preview,
  `use-prepared` and activation are unchanged; `tests/test_training.py` passes as before.
- What it trains (`training/patient_ditto.py`, `configs/patient_ditto.yaml`): starts from the delivered shared
  adapter `models/ditto_llm/sft200-g2-s0`; trains on this patient's examples only, plus 30 anchors (the shared
  adapter's own choices, with the app's option rule, on training situations of other kinds than the examples, as in
  service/live_ditto.py), so the update does not spill over into other situations; brief SFT, then 3 gentle DITTO
  rounds (option scoring, prompt format v2); keeps the best stage on the training rows. No seed demonstrations are
  re-learned.
- The patient adapter is served with the same settings as the shared one (configs/practice.yaml: option scoring,
  prompt format v2); the registry checks the manifest's base model, revision and prompt format.
- The demo rule changed from first-sound to **first-letter** after a failed meaning hint: the delivered shared
  adapter already gives the first-sound hint there (11 of 11 tickable seeded decisions), so first-sound would show
  no change. `scripts/seed_demo_sessions.py` (summary rule; "cat" is skipped while seeding because Whisper mishears
  it) and `scripts/prepare_demo_run.py` (the prepared run) use it; the contract examples still show the original
  first-sound note.
