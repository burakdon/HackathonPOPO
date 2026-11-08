# Handoff: finish the Person 2 backend (CueAgain)

You are implementing the rest of the backend for a hackathon demo. Each of the phases below ends
with **acceptance checks**; we will grade your work against them, so meet each one exactly and
report honestly what passed and what didn't. Work through the phases in order. Each phase must
leave the full test suite green before you start the next.

## 0. Context: read these first, in this order

| file | why |
|---|---|
| `docs/OPEN_DECISIONS.md` | Every product decision (D1–D10, 1b–1d). Do not change a decision; if one blocks you, stop and report it |
| `docs/API_CONTRACT.md` | The spec. §7 (steering) and §8 (training) are what you build; §4–§6 already exist |
| `core/contract.py` | All request/response models plus `rule_matches`, `rule_action` and `summarize`. **Return these models from endpoints; don't invent new response shapes** |
| `service/engine.py` | The practice engine: patients, sessions, decisions, JSON storage under one root. Extend it in the same style |
| `service/patient_api.py` | The FastAPI router for the contract. Add your endpoints here, in the same style |
| `service/app.py` | Where the router and engine are created (search for `Patient practice API`) |
| `tests/test_patient_api.py` | The test style: `create_app(speech_provider=FakeSpeech(), cueagain_root=tmp_path, cueagain_decider=rules_decider)` |
| `scripts/build_contract_examples.py`, `docs/contract_examples/` | Generated sample responses. Regenerate after any contract change |
| `training/ditto.py`, `agents/registry.py` | DITTO training runs and how adapters are found and served |

**What the product does.** A patient names pictures. Each recording is scored automatically
(`speech/scoring.py`). After every counted answer the patient's decision model chooses the next
step, and while tries are left it only chooses *which hint* (decision 1d). A clinician reviews
sessions and steers the model in two ways: by correcting single decisions, or by writing a note
that becomes a rule, which turns into examples. A per-patient DITTO run then trains on those
examples. The demo has to show that the model's decisions change after steering.

**The demo situation.** In `docs/contract_examples/review/GET_session_review.json`, decision
`s-demo-01-d07`: 2 failed tries on "car" with a meaning hint already shown. The model chose the
sentence hint; the clinician's rule asks for `first_sound`.

## Ground rules

- **Always** run with both extras: `uv run --extra llm --extra speech ...`. Plain `uv sync`,
  `uv add` and `uv run` remove the language-model packages. After `uv add`, run
  `uv sync --extra llm --extra speech`.
- **Tests never touch the network or real models.** Inject fakes (decider, rule drafter, trainer,
  speech provider) the way the existing tests do. Real Whisper, Qwen and Claude are only for the
  manual checks at the end of each phase.
- **Additive only.** Don't change existing endpoint behavior or remove fields. If the contract has
  to change, update `docs/API_CONTRACT.md`, `core/contract.py` and the generator together, and list
  the change in your final report.
- **No secrets in git.** The Claude key lives in `.env` (gitignored). Add an empty
  `ANTHROPIC_API_KEY=` line to `.env.example`.
- **Don't commit or push.** We review the diff first.
- **Don't edit `product/`.** That's Person 1's.
- Run lint on the files you create: `uvx ruff check <your files>`.
- Keep claims honest. Anything scripted or illustrative must say so in the data or in a note.

---

## Phase 1: Corrections and examples (contract §7, first part)

**Build:**
1. In the engine, store examples per patient (`Example` from `core/contract.py`), as JSON under
   the engine root (`examples/<patient_id>.json`, or inside the patient record). Keep a counter
   for IDs: `x-0001`, ... per patient, or include the patient ID so they are globally unique.
2. `POST /decisions/{decision_id}/correction` with `CorrectionIn`:
   - Find the stored decision across sessions. Unknown ID → 404.
   - `preferred` must be in that decision's `options` → otherwise 422 with the message from
     `docs/contract_examples/steering/error_422_correction.json`.
   - Creates a pending example (`source: "correction"`). Correcting the same decision again
     replaces its pending example; if that example was already used in training → 409.
   - Returns the `Example`, status 201.
3. Show the correction on the decision in `GET /sessions/{sid}/review`: the `correction` field of
   `StoredDecision`.
4. `GET /patients/{pid}/examples` → `list[Example]`, with `pending` and `used in <run_id>`.
5. `Patient.pending_examples` must reflect the real count.

**Acceptance checks:**
- [ ] Tests: create a session (fake speech + rules), produce a decision, correct it → 201; the
      review shows the correction; the examples list has it; `pending_examples == 1`.
- [ ] A preferred action that isn't an option → 422. An unknown decision → 404. Correcting the same
      decision twice → one example, updated.
- [ ] Examples survive an app restart (same `cueagain_root`).

## Phase 2: Note → rule → matches → examples (§7, second part)

**Build:**
1. A rule drafter in a new file, `service/rule_drafter.py`, behind a small interface
   `draft(note_text) -> (Rule | None, problem: str | None)`, injectable into `create_app`, as the
   decider is. The default implementation uses the **Claude API**:
   - Add the SDK as a main dependency: `uv add anthropic`, then `uv sync --extra llm --extra speech`.
   - Read the key from the environment, loading `.env` first. A `load_dotenv` helper already exists
     in `agents/llm_prompted.py`; reuse or move it. Never send the key to the browser.
   - Model: `claude-opus-5-5` by default, overridable with the env var `CUEAGAIN_RULE_MODEL`.
   - Use structured outputs with a Pydantic model. Documented pattern:
     ```python
     import anthropic
     client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY
     response = client.messages.parse(
         model=model, max_tokens=4000,
         system=SYSTEM_PROMPT,
         messages=[{"role": "user", "content": note_text}],
         output_format=RuleDraft,            # a Pydantic model
     )
     draft = response.parsed_output          # validated RuleDraft instance
     ```
     Check `response.stop_reason` first: on `"refusal"`, return `(None, "...")`.
   - Make `RuleDraft` a **flat** model (no unions): `understood: bool`, `problem: str | None`,
     the optional `when` fields (`last_error`, `min_failed_attempts_on_picture`,
     `max_failed_attempts_on_picture`, `hints_tried_include`, `hints_tried_exclude`,
     `min_missed_pictures_in_a_row`, `level`), `then_op: Literal["retry","next"]`,
     `then_hint: Hint | None`, `then_level: Literal["easier","same","harder"] | None`, and
     `summary: str`. Convert it to the contract's `Rule` yourself and validate it.
     `understood=False`, a failed conversion or any API error → `(None, problem)`.
   - The system prompt explains the rule format (contract §7 table), the 4 hints, and that notes can
     only steer hints and difficulty. It also says that anything else must come back with
     `understood=false` and a short `problem`.
   - No key → `(None, "Rule drafting is not configured (no ANTHROPIC_API_KEY); fill in the rule by hand.")`.
2. `POST /patients/{pid}/notes` with `NoteIn` → `Note`, status 201:
   - `rule_source: "claude"` and `rule_status: "draft"` on success;
   - `rule: null`, `rule_source: "none"`, `rule_status: "none"` and `problem` otherwise.
   - Store notes per patient.
3. `PUT /notes/{note_id}/rule` with `RuleIn` → `Note` with `rule_status: "confirmed"`. Set
   `rule_source: "clinician"` if the rule differs from the draft, or if there was no draft.
4. `GET /notes/{note_id}/matches` → `list[RuleMatch]` over **this patient's** stored decisions,
   only for a confirmed rule (otherwise 409):
   - matching: `rule_matches(rule, decision.context)`;
   - `rule_says = rule_action(rule, ctx)` if it is in the decision's options, else `null` with
     `allowed: false`;
   - `already_agrees` when it equals the decision's action;
   - `situation`: plain English. The generator has a `situation()` helper; move it somewhere
     shared and reuse it;
   - `date` from the decision.
   - Note: `missed_pictures_in_a_row` needs session history. Compute it from the session's
     earlier pictures, or document that it isn't supported yet.
5. `POST /notes/{note_id}/examples` with `ExamplesIn` → `list[Example]` (`source: "note"`,
   `note_id` set, `preferred = rule_says`), status 201. A decision that isn't in this note's
   allowed matches → 422.

**Acceptance checks:**
- [ ] Tests with a **fake drafter**:
      - note → draft;
      - confirm;
      - matches on a session built to contain the demo situation (two failed tries with `semantic`
        shown): the match has `allowed: true`, and a later last-try decision has `allowed: false`;
      - tick → examples appear in `GET /patients/{pid}/examples`.
- [ ] Tests: no key → `rule: null` with the "not configured" problem; drafter returns
      `understood=false` → `rule: null` with its problem; matches on an unconfirmed note → 409.
- [ ] A unit test for the `RuleDraft` → `Rule` conversion, including a relative level
      ("easier").
- [ ] **Manual check with a real key, if one is available:** the demo note
      "When meaning hints have repeatedly failed, try a first-sound hint before moving to another
      picture." gives `hints_tried_include: ["semantic"]`, `failed_attempts_on_picture.min ≥ 2`
      and `then: retry first_sound`. Paste the actual output in your report. If no key is
      available, say so; don't fake it.

## Phase 3: Training runs, preview, activation (§8)

Person 3 owns the training method. You build the job handling around it. Until Person 3 delivers a
"train from the shared adapter" entry point, use what exists: `training/ditto.py` can train a full
run (SFT plus 3 DITTO rounds, about 5–6 minutes on this Mac) on the 20 seed demonstrations plus
extra examples (`--extra-demos FILE`, rows with `scope: "global"`).

**Build:**
1. A trainer interface, injectable like the decider: `start(run_id, patient_id, examples_jsonl_path)`
   and `status(run_id) -> TrainingRun fields`. The default implementation:
   - Writes the patient's examples (all of them, pending and used) to a JSONL file in the
     demonstration row format. Copy the keys from one row of
     `data/demonstrations/demonstrator_a_v1.jsonl`, with `split: "train"`, `scope: "global"`
     and provenance `"clinician example (CueAgain app)"`. The `context` is the decision's context
     **without `patient_id`**. `options` and `preferred` use `core.practice.format_action` strings.
   - Starts `uv run --extra llm --extra speech python -m training.ditto --run-id <run_id>
     --extra-demos <file>` as a **subprocess** (never in the server process: training takes the GPU
     and memory), logging to a file under the engine root.
   - Reads progress from `runs/ditto/<run_id>/manifest.json`: status, the SFT entry, `rounds`.
     Map it to `Progress` (`sft` → `ditto` round k of 3 → `done`).
   - Writes `patient_id` into a small record of yours. Don't rely on the manifest for it.
2. `POST /patients/{pid}/training-runs`:
   - 422 if there are no pending examples; 409 if any run is `queued` or `running`.
   - Run ID: `<pid>-r<n>`.
   - Returns `TrainingRun` (`queued`), status 202.
3. `GET /training-runs/{run_id}` → `TrainingRun`. When the manifest says `completed`:
   - Compute the `preview` once and store it.
   - Mark the examples it used as `used in <run_id>`.
4. **Preview** (contract §8). Compute what you can honestly, and say what is missing:
   - `follows_your_examples`: for each of the run's examples, the action chosen by the new adapter
     (`ditto:<run_id>` through the registry, as in `service/patient_api.model_decider`) and by the
     patient's current model, compared with `preferred`. Label it
     `"trained on these: shows the update took, not evidence"`.
   - `unchanged_elsewhere`: on the 10 **test** situations in the demonstrations file, how often the
     new adapter's choice equals the current model's. Apply the hint-only option rule, so options
     match what the live app offers.
   - `follows_your_rules_new_situations`: for each confirmed rule of this patient, build up to 10
     **new** situations that satisfy it: other pictures, varied history, generated by code from
     the rule's `when` fields. Count how often new and current models pick `rule_action`. If
     you can't do this well, return `total: 0` and say so in the report. **Never invent numbers.**
   - The preview loads the base model plus 2 adapters, so it runs **in the background** after
     completion. Status stays `running` with progress stage `preview` until it's done.
5. `POST /training-runs/{run_id}/use-prepared` with `UsePreparedIn` (contract §10.5):
   - Allowed only if the prepared run is `completed` and has **exactly the same `example_ids`**,
     otherwise 409.
   - Returns the prepared run's result as this run's: `prepared: true`, `prepared_run_id` set,
     status `completed`.
   - The prepared run must be one this engine knows, i.e. created through the same endpoint earlier.
6. `GET /patients/{pid}/models` → `list[ModelVersion]`: `rules`, the shared adapter(s) from
   `registry.versions()`, and this patient's completed runs as `ditto:<run_id>` (`kind: "patient"`),
   with `active` set.
   - **Contract change:** use `ditto:<run_id>` as the version ID for patient adapters, not the
     `SHARED+run` form in the samples. Update the contract text and the generator.
7. `POST /patients/{pid}/active-model` with `ActivateIn`:
   - Only `rules`, an available shared adapter, or this patient's completed runs; otherwise 422.
   - Uses `engine.set_model`, which logs it, and applies to sessions started afterwards. Document
     that sessions already running keep their model.

**Acceptance checks:**
- [ ] Tests with a **fake trainer** (it writes a manifest that goes queued → running → completed)
      and a **fake decider/preview**:
      - start → 202;
      - a second start while running → 409;
      - no pending examples → 422;
      - completed → examples are `used in <run_id>` and the preview has counts with totals;
      - use-prepared with equal example IDs → `prepared: true`, with different ones → 409;
      - the models list includes the run; activating it changes `GET /patients/{pid}` and the next
        session's `model_version`.
- [ ] **Manual check:** one real run for `demo-ana`, with the 2–3 examples from Phase 2. Report its
      duration, its status, and the real preview numbers, whatever they are. If it fails, report
      the error from the manifest.

## Phase 4: Hint audio (D7)

**Build:** `scripts/make_hint_audio.py`. For each of the 24 pictures and each clip
(`first_sound`, `first_letter`, `sentence_frame`, `semantic`, `whole_word`):
- Generate the audio with macOS `say -v Samantha -o tmp.aiff "<text>"` and convert it with
  `afconvert -f m4af -d aac tmp.aiff frontend/hints/<exercise_id>/<clip>.m4a`.
- Text: the cue text from `data/items.json`. Blank markers like `___` are spoken as "blank".
  `whole_word` is the word itself.
- Skip clips that already exist unless `--force`, so recordings dropped in by hand survive.
- Write `frontend/hints/README.md`: how to replace a clip, and that first-sound clips should be
  re-recorded by a person.

**Acceptance checks:**
- [ ] 120 files exist, and `GET /pictures` then returns non-null `audio_url` values; check with the
      real app.
- [ ] Re-running doesn't overwrite existing files; `--force` does.

## Phase 5: Scripted demo history (§10.1)

**Build:** `scripts/seed_demo_sessions.py`. It fills `demo-ana` with 3 sessions played through the
**real** pipeline:
- Mac-voice clips go through real Whisper and the real scorer, via the app in-process
  (`fastapi.testclient.TestClient(create_app())`).
- Start each session with `engine.start(..., simulated=True)`, so the sessions are marked
  simulated.
- Use the patient's current real model.
- Script the answers so that "a meaning hint failed twice" happens at least 4 times, because
  that's what the demo note is about.
- Include a few correct answers, sound errors and one "I'm not sure".
- Idempotent: `--reset` removes only the sessions this script created (keep a list of their IDs).
- Print a summary: per session, the counts and how many decisions match the demo rule.

**Acceptance checks:**
- [ ] After running it, the demo rule's matches return at least 4 tickable decisions with
      recordings that play back.
- [ ] Every session is `simulated: true`, and the transcripts are what Whisper really heard. If
      Whisper mishears a clip, keep the real result and adjust the script's word choice; never
      write a fake transcript.

## Phase 6: Scorer accuracy on synthetic clips (D9)

**Build:** `scripts/scorer_report.py`.
- Generate labeled clips for every picture with at least 3 macOS voices (pick from `say -v '?'`;
  English voices only).
- Categories, each with its expected outcome:
  - correct word;
  - accepted alternative (where the picture has one);
  - plural;
  - wrong real word (expected meaning error);
  - near-miss, changing the first sound (expected sound error);
  - "it's a <word>" (expected unclear);
  - filler + word ("um, <word>", expected correct);
  - "I don't know" (expected no answer);
  - silence (a generated silent WAV, expected unclear).
- Run every clip through `WhisperLocalProvider` and `speech/scoring.score`.
- Write `results/scorer_synthetic.md` and a CSV containing:
  - a confusion table, expected vs actual;
  - error-type accuracy for incorrect answers;
  - **the headline: how often an answer that should be incorrect came back "correct"**, with
    examples;
  - latency;
  - the caveat, written out: synthetic speech is far cleaner than real or aphasic speech, so these
    numbers are optimistic and say nothing about patients.
- **Don't change the scorer's thresholds.** If the data suggests a change, propose it in the report
  with the evidence.

**Acceptance checks:**
- [ ] The report exists, its numbers come from an actual run, and its CSV lists every clip with
      what was said, what was heard, and the expected and actual outcome.

## Phase 7: Small fixes

- [ ] The old practice router (`service/practice.py`, `restore()`) prints
      `could not restore active_model.json`. Make `restore()` skip files that aren't session
      snapshots, without changing anything else.
- [ ] Add `ANTHROPIC_API_KEY=` and `CUEAGAIN_RULE_MODEL=` (commented) to `.env.example`.

---

## Finishing

1. `uv run --extra llm --extra speech python -m scripts.build_contract_examples`, then
   `uv run --extra llm --extra speech pytest -q`: everything passes, and the new tests are included.
2. `uvx ruff check` on every file you created: clean.
3. Update `docs/API_CONTRACT.md` for every contract change, and add an "Implemented" note under
   the relevant decisions in `docs/OPEN_DECISIONS.md`.
4. Write your report in `docs/HANDOFF_PERSON2_REPORT.md`:
   - per phase: done / partial / not done, and every acceptance check with pass/fail;
   - the manual checks' actual outputs;
   - every contract change;
   - anything you weren't sure about, or decided yourself;
   - known gaps.
5. Don't commit.

## How we'll review

We'll read the diff, run the tests, and spot-check:
- the error cases (404, 409, 422);
- that nothing scripted or illustrative is presented as measured;
- that no secret is in the repository;
- that the demo situation goes all the way through: rule → match → example → training run →
  preview → activation → a new session where the model gives `first_sound` in that situation.
