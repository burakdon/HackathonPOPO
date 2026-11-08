# CueAgain API contract (draft for review)

**Status:** draft v0.1, 2026-11-07. Persons 1 and 3: review and comment before Person 2 builds.

This document is how the parts talk to each other:
- the frontend (`product/`, Person 1),
- the backend and scoring (Person 2),
- the decision model (Person 3).

It follows the decisions in [OPEN_DECISIONS.md](OPEN_DECISIONS.md), referred to as D1 to D10 below.
Field names follow `product/mock/data.json` wherever the mock has them, so Person 1 can replace the
mock with few changes.

Existing endpoints stay unchanged:
- `/health`, `/act`, `/update`, `/override`, `/learn_from_overrides`;
- the current `/practice/...` endpoints, used by the developer console in `frontend/`.

Everything below is new.

## Contents
1. [Conventions and shared values](#1-conventions-and-shared-values)
2. [Pictures and hint audio](#2-pictures-and-hint-audio)
3. [Patients and plans](#3-patients-and-plans)
4. [Patient practice](#4-patient-practice)
5. [The decision model interface (Person 3)](#5-the-decision-model-interface-person-3)
6. [Clinician review](#6-clinician-review)
7. [Clinician steering: corrections and notes](#7-clinician-steering-corrections-and-notes)
8. [Model updates](#8-model-updates)
9. [Who builds what](#9-who-builds-what)
10. [Resolved review questions](#10-resolved-review-questions-kuba-2026-11-07)

---

## 1. Conventions and shared values

- JSON everywhere except audio uploads (multipart form) and audio downloads (files).
- IDs are opaque strings. Times are ISO 8601 in UTC.
- Errors are `{"detail": "..."}` with these status codes:

  | code | meaning |
  |---|---|
  | 404 | unknown ID |
  | 409 | stale or conflicting request |
  | 422 | invalid input |
  | 503 | speech recognition or the model is unavailable |

- **Idempotency:** every request that records something carries a `client_attempt_id` (a UUID made
  by the browser).
  - Sending the same ID again returns the original result and records nothing new.
  - The same ID with different content returns 409.
  - So a retry after a dropped connection is always safe.

| name | values | notes |
|---|---|---|
| `outcome` | `correct` · `incorrect` · `unclear` | D5, D6. `unclear` is never counted |
| `error_type` | `sound` · `meaning` · `no_response` · `unclassified` | Only when `incorrect`. The UI shows `meaning` as "Wrong word" (as in the mock) |
| `hint` | `first_sound` · `first_letter` · `sentence_frame` · `semantic` | The model can choose these |
| `whole_word` | — | An audio clip only. Played by the fixed "show the word" step after the last try, never chosen by the model |
| `level` | `1` · `2` · `3` | 1 is easiest. The mock and the UI say "level"; the existing code says "tier" |
| picture `result` | `independent` · `assisted` · `unsuccessful` · `not_assessed` | Section 6 |

## 2. Pictures and hint audio

### `GET /pictures`
Returns all 24 pictures with their hints and audio.

```json
[
  {
    "exercise_id": "ex-cat",
    "word": "cat",
    "accepted": ["kitty", "kitten"],
    "level": 1,
    "image_url": "/media/images/cat.svg",
    "hints": {
      "first_sound":    {"text": "It starts with the sound “k…”", "audio_url": "/media/hints/ex-cat/first_sound.m4a"},
      "first_letter":   {"text": "It starts with the letter C.",   "audio_url": "/media/hints/ex-cat/first_letter.m4a"},
      "sentence_frame": {"text": "A pet that says meow is a ___.", "audio_url": "/media/hints/ex-cat/sentence_frame.m4a"},
      "semantic":       {"text": "A small furry pet that purrs.",  "audio_url": "/media/hints/ex-cat/semantic.m4a"}
    },
    "whole_word_audio_url": "/media/hints/ex-cat/whole_word.m4a",
    "attribution": "Twemoji 15.1.0, CC BY 4.0"
  }
]
```

- **Audio files:** `/media/hints/<exercise_id>/<hint>.m4a`. AAC plays in Chrome and Safari.
- They are generated once with macOS `say` (D7). To use a real recording, put a file with the same
  name in its place.
- `audio_url` is `null` while a clip has not been generated. The UI then shows only the text.

## 3. Patients and plans

The server stores patients and plans (D3). Plan field names match the mock's `plan`.

### `GET /patients`
Returns the caseload: one row per patient, with the summary fields of their latest session (§6).

### `GET /patients/{patient_id}`
```json
{
  "id": "maria",
  "name": "Maria Alvarez",
  "first": "Maria",
  "simulated": false,
  "about": "Retired teacher.",
  "plan": {
    "level": 1,
    "pictures_per_session": 12,
    "max_attempts": 3,
    "break_after_misses": 3,
    "hint_order": "match_error",
    "session_minutes": 15,
    "days_per_week": 5,
    "personal_words": ["Sofia", "Mateo"]
  },
  "active_model": {"version": "ditto:demoA-qwen3-1.7b-r1", "patient_adapter": null},
  "pending_examples": 3
}
```

**Plan fields:**
- `level`: the starting level of each session.
- `max_attempts`: scored tries per picture before "show the word".
- `break_after_misses`: unsuccessful pictures in a row before a break is suggested.
- `hint_order`: used **only** as the fallback when the model's output is invalid (D1). Values as in
  the mock: `match_error` · `first_sound_first` · `meaning_first` · `least_help`.
- `personal_words`: stored and shown, but **not used** yet, because those words have no pictures.

### `POST /patients`
Body: `{"name": "...", "first": "...", "about": "..."}`. Returns the new patient with the default
plan.

### `PUT /patients/{patient_id}/plan`
- Body: a full plan. Returns the saved plan.
- Applies from the patient's next session.
- Each change is written to the patient's change log, together with `who` (free text, for example
  "Dr. Patel, SLP").

### `GET /patients/{patient_id}/log`
Returns the change log: plan changes, confirmed rules, training runs and model activations.
```json
[{"when": "2026-11-07T15:02:00Z", "who": "Dr. Patel, SLP", "what": "Tries before showing the word: 3 → 4."}]
```

## 4. Patient practice

### The loop
1. Show a picture.
2. The patient records an answer.
3. The server scores it and returns the next step.

The frontend never decides what comes next. It shows what `next` says.

```
picture ─► record ─► POST /recordings ─► next.kind
                                          ├─ correct       → "Yes! cat."    → [Next] → next.picture
                                          ├─ hint          → hint text + audio → record again (same picture)
                                          ├─ show_word     → "This is a cat. Say it with me." + whole-word audio → [Next] → next.picture
                                          ├─ try_again     → "I didn't catch that." → record again (not counted)
                                          └─ not_assessed  → "Let's try another one." → next.picture
           [I'm not sure] ─► POST /no-answer ─► same as above
```

If `next.session_complete` is true, there is no `next.picture`. The UI shows the end screen using
`next.summary`.

### `POST /patients/{patient_id}/sessions`
Starts a session. The body is optional: `{"device": "tablet"}`.

```json
{
  "session_id": "s-7f3a",
  "patient_id": "maria",
  "model_version": "ditto:demoA-qwen3-1.7b-r1",
  "plan": { "...": "a copy of the plan used for this session" },
  "picture": {
    "presentation_id": "p-01",
    "number": 1,
    "of": 12,
    "exercise_id": "ex-cat",
    "word": "cat",
    "level": 1,
    "image_url": "/media/images/cat.svg",
    "attempt_number": 1,
    "max_attempts": 3,
    "hint": null
  }
}
```

- `word` is included for the "Yes! cat." and "show the word" screens. The UI must not show it
  earlier.
- The first picture is chosen at `plan.level`.

### `POST /sessions/{session_id}/recordings`
A multipart form with these fields:

| field | type | |
|---|---|---|
| `audio` | file | WebM/Opus (Chrome) or MP4/AAC (Safari), as recorded by the browser |
| `presentation_id` | string | the picture on screen. If it is no longer the current one: 409 |
| `client_attempt_id` | string | idempotency key (§1) |

**Response: the result of the attempt and the next step.**
```json
{
  "attempt_id": "a-0003",
  "presentation_id": "p-01",
  "counted": true,
  "transcript": "uuummm tat",
  "score": {
    "outcome": "incorrect",
    "error_type": "sound",
    "heard_word": "tat",
    "matched_name": null,
    "reason": "One word, “tat”, sounds close to “cat” (differs in the first sound).",
    "signals": {"audio_seconds": 1.8, "no_speech_prob": 0.04, "word_confidence": 0.71}
  },
  "next": {
    "kind": "hint",
    "message": "Here's a hint.",
    "hint": {"type": "first_sound", "text": "It starts with the sound “k…”", "audio_url": "/media/hints/ex-cat/first_sound.m4a"},
    "attempt_number": 2,
    "max_attempts": 3,
    "decision_id": "d-0002"
  }
}
```

**How `next.kind` is chosen** (fixed rules, D1 and D6):

| score | `counted` | `next.kind` | who decides the content |
|---|---|---|---|
| `correct` | yes | `correct` | the model chooses the next picture's level |
| `incorrect`, tries left | yes | `hint` *or* the model moves on | the model: retry with a hint, or the next picture at a level |
| `incorrect`, last try | yes | `show_word` | fixed. Then the model chooses the next level |
| `unclear`, 1st in a row | no | `try_again` | fixed |
| `unclear`, 2nd in a row | no | `not_assessed` | fixed, at the same level. The model isn't asked |

- When the model chooses "next picture" after an incorrect answer that still had tries left,
  `next.kind` is `move_on` and carries `next.picture`.
- **Every response that leaves the current picture** (`correct`, `show_word`, `not_assessed`,
  `move_on`) carries:
  - `next.picture`, the next presentation, or `next.session_complete: true` with `next.summary`;
  - `next.break_suggested`, true when `break_after_misses` unsuccessful pictures in a row have been
    reached. The UI then shows the break screen before the next picture.
- `decision_id` is present whenever the model was asked, so the clinician can review that decision
  later (§6).

**Scoring rules** (D4, D5, D6). They are applied to the transcript, which is never rewritten.

**1. Unclear first.** Any of these makes the attempt `unclear`:
- no speech;
- shorter than the minimum length;
- low recognition confidence;
- two or more words left after fillers are removed. "it's a cat" is unclear.

**2. Fillers are removed.** These go, however they are drawn out: a, an, the, um, uh, er, erm, hmm.
So "uuummm cat" is a single word, "cat".

**3. Correct:** the single word is the target or an accepted name, or their plural, or it sounds
exactly the same (CMUdict).

**4. Incorrect:** any other single word. The error type is:
- `sound` when the word sounds close to the target;
- `meaning` when it is a different real word;
- `unclassified` otherwise.

`reason` is a plain-English sentence the clinician view can show as it is.

### `POST /sessions/{session_id}/no-answer`
The "I'm not sure" button. Body: `{"presentation_id": "p-01", "client_attempt_id": "..."}`.
- Same response as a recording.
- `outcome: incorrect`, `error_type: no_response`, `transcript: null`, no audio.
- It counts as a try.

### `POST /sessions/{session_id}/end`
Ends the session early: Pause → Stop, or the patient closes the tablet. Returns the summary (§6).
Sessions also end on their own after `pictures_per_session` pictures.

### Errors during practice
| situation | response |
|---|---|
| Speech recognition is unavailable | 503 `{"detail": "speech recognition unavailable"}`. The UI says practice can't continue right now. Nothing is recorded |
| Recording for a picture that is no longer current | 409. The UI reloads `GET /sessions/{id}` and shows the current picture |
| The model times out or returns an invalid action | not an error. The plan's `hint_order` fallback decides, and the decision is recorded with `source: "fallback"` and the reason |

### `GET /sessions/{session_id}`
Returns the current state of the session: the picture on screen, its attempt number and hint, and
whether the session is complete. Used to recover after a reload.

## 5. The decision model interface (Person 3)

This is a Python interface, not HTTP. The server calls it after every counted attempt.

```python
decide(context: DecisionContext, options: list[PracticeAction], patient_id: str) -> Decision
```

**`context`:** what the model sees. It is the existing `core/practice.py` `DecisionContext`,
extended with `patient_id`, and its attempts now come from automatic scoring.
```json
{
  "patient_id": "maria",
  "current": {"exercise_id": "ex-cat", "word": "cat", "tier": 1, "hints_shown": ["semantic"], "attempts": 2},
  "history": [
    {"exercise_id": "ex-dog", "word": "dog", "tier": 1, "hint": "none",     "correct": true,  "error": null,      "response_time_s": 2.1},
    {"exercise_id": "ex-cat", "word": "cat", "tier": 1, "hint": "none",     "correct": false, "error": "meaning", "response_time_s": 3.4},
    {"exercise_id": "ex-cat", "word": "cat", "tier": 1, "hint": "semantic", "correct": false, "error": "meaning", "response_time_s": 4.0}
  ],
  "session_minutes": 3.2
}
```
- `history` holds counted attempts only. Unclear recordings are never included.
- `error` uses `error_type` values.

**`options`:** every allowed action. The server builds the list from the plan, so the model can't
break it:
- `{"op":"retry","hint":"first_sound"}` and the other hints not yet shown on this picture, but only
  after an incorrect answer and only while tries are left;
- `{"op":"next","tier":N}` for levels within 1 of the current one.

**`Decision`:**
```json
{
  "action": {"op": "retry", "hint": "first_sound"},
  "source": "ditto:demoA-qwen3-1.7b-r1+maria-r2",
  "fallback_reason": null,
  "raw_output": "{\"op\":\"retry\",\"hint\":\"first_sound\"}",
  "latency_ms": 570
}
```
- If the output isn't one of the options, or takes longer than `timeout_s`, the server uses the
  fallback. It then records `source: "fallback"` and `fallback_reason`, for example
  `"invalid output"` or `"timeout"`.

**Stored decision.** Every decision is stored, including the full context and options, for
clinician review and training examples:
```json
{
  "decision_id": "d-0002",
  "session_id": "s-7f3a",
  "presentation_id": "p-01",
  "after_attempt_id": "a-0003",
  "context": { "...": "as above" },
  "options": [{"op":"retry","hint":"first_sound"}, {"op":"retry","hint":"first_letter"}, {"op":"next","tier":1}, {"op":"next","tier":2}],
  "action": {"op": "retry", "hint": "first_sound"},
  "source": "ditto:demoA-qwen3-1.7b-r1+maria-r2",
  "fallback_reason": null,
  "latency_ms": 570,
  "at": "2026-11-07T15:04:11Z"
}
```

## 6. Clinician review

The clinician can play recordings and see transcripts and results, but cannot change them (D4).

### The three summaries (per picture, never per recording)
| field | definition |
|---|---|
| `named_without_hint` | named correctly on the first counted try, with no hint (independent) |
| `named_with_hint` | named correctly after at least one hint (assisted) |
| `unsuccessful` | the picture ended without a correct answer, through "show the word" or the model moving on |
| `not_assessed` | two unclear recordings in a row. Not in the denominator |

- The first three are fractions of the assessed pictures, as in the mock.
- `counts` gives the raw numbers.

### `GET /patients/{patient_id}/sessions`
Returns one row per session, oldest first, in the same shape as the mock's `sessions` (without
`trials`):
```json
[{
  "session_id": "s-7f3a",
  "date": "2026-11-07",
  "simulated": false,
  "model_version": "ditto:demoA-qwen3-1.7b-r1",
  "pictures": 12,
  "named_without_hint": 0.42,
  "named_with_hint": 0.25,
  "unsuccessful": 0.33,
  "counts": {"assessed": 12, "independent": 5, "assisted": 3, "unsuccessful": 4, "not_assessed": 1},
  "errors": {"sound": 6, "wrong_word": 3, "no_answer": 1, "unclassified": 0},
  "mean_level": 1.3,
  "minutes": 9.5,
  "ended_early": false
}]
```
`errors` counts error types over all counted incorrect attempts, using the mock's key names.

### `GET /sessions/{session_id}/review`
Returns everything about one session, picture by picture:
```json
{
  "session_id": "s-7f3a",
  "summary": { "...": "the row above" },
  "pictures": [{
    "presentation_id": "p-01",
    "exercise_id": "ex-cat",
    "word": "cat",
    "level": 1,
    "image_url": "/media/images/cat.svg",
    "result": "assisted",
    "attempts": [
      {"attempt_id": "a-0001", "number": 1, "hint_shown": null,       "transcript": "dog",
       "outcome": "incorrect", "error_type": "meaning", "counted": true,
       "reason": "One word, “dog”, a different word from “cat”.", "audio_url": "/attempts/a-0001/audio", "at": "..."},
      {"attempt_id": "a-0002", "number": null, "hint_shown": "semantic", "transcript": "",
       "outcome": "unclear", "error_type": null, "counted": false,
       "reason": "No speech detected.", "audio_url": "/attempts/a-0002/audio", "at": "..."},
      {"attempt_id": "a-0003", "number": 2, "hint_shown": "semantic", "transcript": "cat",
       "outcome": "correct", "error_type": null, "counted": true, "audio_url": "/attempts/a-0003/audio", "at": "..."}
    ],
    "decisions": [{ "...": "stored decisions on this picture (§5), each with its correction if any (§7)" }]
  }]
}
```

The mock's `trials` table has one row per picture, where the hint belongs to the *next* picture.
That doesn't fit the new flow (D1). Show `pictures`, with a decision row after each attempt the
model reacted to. The "I'd do this differently" control goes on decision rows.

### `GET /attempts/{attempt_id}/audio`
Returns the patient's recording in its original format. Audio is kept under git-ignored `runs/`
(D8).

## 7. Clinician steering: corrections and notes

Both lead to **training examples**: a stored decision's context, plus the action the clinician
prefers. Examples stay **pending** until a training run uses them (§8).

### Correcting one decision: `POST /decisions/{decision_id}/correction`
```json
{"preferred": {"op": "retry", "hint": "first_sound"}, "note": "For this patient, try a sound cue before another meaning cue."}
```
- `preferred` must be one of the decision's `options`, otherwise 422.
- The note is optional. It is stored as context and is not trained on.
- Returns the new example:
```json
{"example_id": "x-0001", "patient_id": "maria", "source": "correction", "decision_id": "d-0002",
 "model_chose": {"op":"retry","hint":"semantic"}, "preferred": {"op":"retry","hint":"first_sound"},
 "note": "For this patient, try a sound cue before another meaning cue.", "status": "pending"}
```
Correcting the same decision again replaces the pending example. If that example was already used
in training, the response is 409.

### A note becomes a rule, then examples (D1b)

**Step 1: `POST /patients/{patient_id}/notes`.** The clinician writes a note, and the server drafts a
rule with the Claude API. The key stays on the server.
```json
{"text": "When meaning hints have repeatedly failed, try a first-sound hint before moving to another picture."}
```
Response:
```json
{
  "note_id": "n-0001",
  "text": "When meaning hints have repeatedly failed, ...",
  "rule": {
    "when": {"hints_tried_include": ["semantic"], "failed_attempts_on_picture": {"min": 2}, "last_error": ["meaning", "sound", "no_response", "unclassified"]},
    "then": {"op": "retry", "hint": "first_sound"},
    "summary": "After 2 or more failed tries on a picture where a meaning hint was already given → give the first-sound hint."
  },
  "rule_source": "claude",
  "rule_status": "draft",
  "problem": null
}
```

**Rule format.** Every part of `when` is optional, and all the given parts must hold.

| `when` field | meaning |
|---|---|
| `last_error` | the latest counted attempt's error type is one of these. `["correct"]` means it was correct |
| `failed_attempts_on_picture` | `{"min": n}` and/or `{"max": n}` |
| `hints_tried_include` | these hints were already shown on this picture |
| `hints_tried_exclude` | these hints were not shown |
| `missed_pictures_in_a_row` | `{"min": n}` |
| `level` | the current level is one of these |

`then` is either:
- `{"op":"retry","hint":H}`, or
- `{"op":"next","level":"easier"|"same"|"harder"}`. This is relative, because the clinician steers
  direction only (D10b).

**When drafting fails.** If there is no API key, Claude's output doesn't validate, or the note asks
for something outside the format:
- `rule` is `null`;
- `rule_source` is `"none"`;
- `problem` explains why, for example "Notes can steer hints and difficulty only".

The UI then shows a "when ___ → do ___" form built from the table above. A rule is never applied
without the clinician confirming it.

**Step 2: `PUT /notes/{note_id}/rule`.** The clinician confirms the rule, as drafted or edited. Body:
`{"rule": {...}}`. It is checked against the format and returns `rule_status: "confirmed"`.

**Step 3: `GET /notes/{note_id}/matches`.** Returns this patient's stored decisions that match the
confirmed rule:
```json
[{"decision_id": "d-0002", "date": "2026-11-07", "word": "cat",
  "situation": "2 failed tries on “cat”, meaning hint given, last answer a wrong word",
  "model_chose": {"op":"retry","hint":"semantic"}, "rule_says": {"op":"retry","hint":"first_sound"},
  "already_agrees": false, "allowed": true}]
```
- `allowed: false` means the rule's action wasn't one of that decision's options, for example no
  tries left. Those can't be ticked.
- A relative `level` is turned into an absolute one for each match.

**Step 4: `POST /notes/{note_id}/examples`.** Body: `{"decision_ids": ["d-0002", "d-0007"]}`.
Creates one pending example per ticked decision, with `source: "note"` and `note_id`.

### `GET /patients/{patient_id}/examples`
Returns all examples, with `status: "pending"` or `"used in maria-r2"`. The UI must keep these two
states visibly separate.

## 8. Model updates

One shared starting adapter, plus one adapter per patient trained from it (D1c). Training runs in the
background, about 5 minutes on the laptop. One run at a time.

### `POST /patients/{patient_id}/training-runs`
- Starts training on all of the patient's pending examples, plus any earlier ones.
- Returns `{"run_id": "maria-r2", "status": "queued", "example_ids": [...]}`.
- 422 if there are no pending examples. 409 if a run is already in progress.

### `GET /training-runs/{run_id}`
```json
{
  "run_id": "maria-r2",
  "patient_id": "maria",
  "status": "running",
  "progress": {"stage": "ditto", "round": 2, "of": 3},
  "started": "...", "finished": null,
  "base": "ditto:demoA-qwen3-1.7b-r1",
  "example_ids": ["x-0001", "x-0002", "x-0003"],
  "preview": null,
  "error": null
}
```
`status` is one of `queued` · `running` · `completed` · `failed`.

When the run is `completed`, `preview` compares the new adapter with the patient's active model:
```json
{
  "follows_your_examples": {"new": 3, "current": 0, "total": 3, "label": "trained on these: shows the update took, not evidence"},
  "follows_your_rules_new_situations": {"new": 7, "current": 1, "total": 10},
  "unchanged_elsewhere": {"same": 9, "total": 10}
}
```
- `follows_your_rules_new_situations`: situations the model hasn't seen that match the patient's
  confirmed rules. How they are generated is for Person 3 to define.
- `unchanged_elsewhere`: the general held-out test situations where the new adapter's choice equals
  the current one's.
- Every number is a count with its total.

### `POST /patients/{patient_id}/active-model`
- Body: `{"version": "ditto:demoA-qwen3-1.7b-r1+maria-r2"}`, or the shared version to undo.
- Only `completed` runs can be activated.
- Applies from the next decision. Logged in the change log.

### `GET /patients/{patient_id}/models`
Returns the versions available to this patient: bandit, shared adapter, and this patient's adapters,
with which one is active.

## 9. Who builds what

| part | owner |
|---|---|
| §2 pictures and hint audio, §3 patients and plans, §4 practice loop and scoring, §6 review, §7 corrections and notes (including the Claude call), the storage and endpoints of §8 | Person 2 |
| The scoring accuracy report on synthetic clips (D9) | Person 2 |
| `decide()` (§5), per-patient DITTO training started from §8, the `preview` numbers, the rule-matching held-out situations | Person 3 |
| `product/` wired to these endpoints: tablet loop, recording, hint audio, review with playback, corrections, notes → rule → ticking matches, training progress, preview, activation | Person 1 |

**Until the backend exists,** Person 2 commits example responses for every endpoint in
`docs/contract_examples/`, so Person 1 can build against them.

## 10. Resolved review questions (Kuba, 2026-11-07)

This is a hackathon demo. The goal is to show that the clinician's steering visibly changes the
model's decisions. The resolutions below favor that, and favor prepared files over features.

**1. Demo history: scripted sessions through the real pipeline.**
- A script, `scripts/seed_demo_sessions.py` (Person 2), plays synthetic answers through the real
  endpoints for one demo patient (`simulated: true`): Mac-voice clips of correct words, wrong words,
  near-misses and silence.
- This produces several sessions with real transcripts, scores, playable audio and stored model
  decisions, so the clinician has something to steer on.
- The answers are scripted so that "meaning hint failed" situations come up often, which is the
  case the demo note is about.
- Maria's and James's mock histories stay display-only in the charts. They can't be corrected,
  because they have no stored decisions.

**2. "Show the word": not recorded.** The patient presses "I said it. Next picture", as in the mock.

**3. Hints already shown are never offered again on the same picture.** Person 3 should check that
the existing training examples still fit.

**4. The model is always asked, including after a correct answer.** Difficulty can only be steered
through stored decisions.

**5. Training in the demo: started live, result prepared.**
- In the demo, the clinician writes the note, confirms the rule, ticks examples and presses Train.
  Training really starts (§8).
- While it runs, we switch to an identical run trained beforehand on the same examples. Then we show
  its preview numbers and a new session where the model gives the first-sound hint.
- We say plainly that the result shown was prepared earlier.
- To support this: `POST /training-runs/{run_id}/use-prepared {"prepared_run_id": "..."}` is
  accepted only when the prepared run's `example_ids` equal the live run's. Its response says
  `prepared: true`, and the UI shows a "prepared earlier" label.
