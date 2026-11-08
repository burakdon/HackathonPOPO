# Open decisions before we build

Decide these in order: later items depend on earlier ones. For each one, record the decision, who
decided it, and the date. "Proposed" is a suggestion, not a decision.

Status: `open` · `decided` · `deferred`

| # | Topic | Who decides | Status |
|---|---|---|---|
| 1 | Practice flow: retry the same picture, and what chooses the hint | Team | decided |
| 1b | How a clinician's note becomes training data | Team (Persons 2 and 3) | decided |
| 1c | One shared model, or a separate adapter per patient | Person 3 | decided |
| 1d | Hint only while tries are left | Team | decided |
| 2 | Which UI we demo (`product/` or `frontend/`) | Team | decided |
| 3 | Where patients and practice plans live | Team (Persons 1 and 2) | decided |
| 4 | Automatic error type: sound error / wrong word / no answer | Team (Persons 2 and 3) | decided |
| 5 | When an answer counts as correct | Person 2 | decided |
| 6 | Unclear recordings: when to ask again, and when to stop | Person 2 | decided |
| 7 | Hint audio: source and format | Person 2 | decided |
| 8 | Storing recordings, and which speech model to use | Person 2 | decided |
| 9 | Labeled test recordings to measure the scorer | Team | decided |
| 10 | Data formats between the frontend, backend and model | Team | decided |

---

## 1. Practice flow: retry the same picture, and what chooses the hint

**Why it matters:** the pieces disagree today.
- The `product/` tablet retries the same picture and picks hints from the plan's fixed hint order.
- The `product/` clinician table and "Teach the AI" assume the older setup: one row per picture,
  where the hint is for the *next* picture after an error.
- The spec retries the same picture, and the model chooses the hint.
- `dev` has two training loops: PPO `/learn_from_overrides` (seconds, CPU) and the language-model
  DITTO (about 5 minutes on a laptop GPU).

**Proposed:**
- Retry the same picture everywhere.
- The plan's hint order is the baseline that chooses hints, behind one server call. Person 3's
  model can replace it.
- Every hint decision is stored separately, so the clinician corrects individual decisions, not
  whole pictures.
- "Teach the AI" trains whichever model is behind that call.

**Decision (Kuba, 2026-11-07):**
- Retry the same picture.
- The model decides each step: which hint to give, or whether to move to another picture. The
  current action set already covers this: "retry with hint X" or "next picture at level N".
- The process is customized per patient. The clinician steers the model after seeing a patient's
  tries, both by correcting individual decisions and by writing notes such as "When meaning hints
  have repeatedly failed, try a first-sound hint before moving to another picture."
- DITTO turns that steering into a model update.
- The plan's fixed hint order stays only as the fallback when the model's output is invalid.

**Feasibility (assessed 2026-11-07):** feasible, with three caveats.
- DITTO learns from examples (situation → preferred action), not from text. A note has to become
  examples first (see 1b).
- With a handful of examples per patient, the effect must be measured. Report how often the model
  follows the rule on situations it hasn't seen, and how often its other decisions stay unchanged.
- The current adapter misses every "retry with a hint" situation, which is exactly this case. The
  training data needs more of them.

### 1d. Hint only while tries are left

**Decision (Kuba, 2026-11-07): Option A.** After a wrong answer with tries left, the model is offered
hints only. It chooses *which* hint, never whether to move on early. Moving on happens after the last
try ("say it with me") or after a correct answer, and the model then chooses the next level.

**Why:** in a live spoken session, the DITTO adapter `demoA-qwen3-1.7b-r1` moved to a new picture after
every wrong answer (4 of 4), so patients never got a hint. That broke the spec ("if incorrect,
provide a hint and allow another attempt"). The steering the demo shows is a choice between hints,
which the model still makes.

**Trade-off:** the model can't learn "move on early when it's too hard". If a retrained adapter
handles that well, relax this to "at least one hint per picture" (Option C), a one-line change in
`service/engine.py`.

### 1b. How a clinician's note becomes training data

**Decision (Kuba, 2026-11-07): note → rule → examples.**
1. A language model turns the note into a structured rule, for example "after 2 or more failed
   meaning hints on the same picture → give the first-sound hint". The clinician confirms or edits
   it.
2. The app finds moments in this patient's past sessions that match the rule, and shows what the
   model did next to what the rule asks.
3. The clinician ticks which ones to keep. The ticked ones become DITTO examples, together with any
   decisions the clinician corrected directly.

The note is stored with the rule as context. It is not put in the model's prompt.

**Rule drafting (Kuba, 2026-11-07): the Claude API, built by Person 2.**
- The server sends the note and the list of allowed situations and actions to Claude, and gets back
  a structured rule that is checked against that list.
- The API key stays in the server's `.env` and is never committed or sent to the browser.
- Without a key, or when the output doesn't validate, the clinician fills in a "when ___ → do ___"
  form instead. The note is never turned into a rule silently.

### 1c. One shared model, or a separate adapter per patient

**Decision (Kuba, 2026-11-07): a shared starting adapter plus one adapter per patient.**
- The shared adapter is trained on the general examples, currently the 20 training situations.
- Each patient's update trains that patient's own adapter from the shared one, using only that
  patient's examples. Steering one patient never changes another.
- Each adapter is tens of MB and takes about 5 minutes to train on the laptop.

## 2. Which UI we demo

**Proposed:** `product/` becomes the demo UI, served by the server, with its mock data replaced by
server calls (Person 1). `frontend/` becomes a developer console or is removed. Simulated patients
stay, clearly labeled, next to real sessions.

**Decision (Kuba, 2026-11-07): as proposed.**
- `product/` is the demo UI, served by the server.
- Person 1 replaces the mock data with server calls and removes the helper panel.
- `frontend/` is kept only as a developer console for testing the backend.
- Simulated patients stay, labeled, next to real sessions.

## 3. Where patients and practice plans live

**Why it matters:** the server has no patients or plans. A session only has an optional free-text
patient label. The prototype keeps patients and plans in its mock data.

**Proposed:** the server stores them (Person 2): patient ID, name, a "simulated" flag, and the plan
(level, hint order, pictures per session, tries before showing the word, break after misses).

**Decision (Kuba, 2026-11-07): the server stores them (Person 2).**
- Patients and their plans are saved as JSON on the server.
- A patient has an ID, a name and a "simulated" flag.
- A plan has a starting level, pictures per session, tries before showing the word, break after
  misses, and a fallback hint order.
- Sessions, training examples and adapters are linked to a patient ID.
- The plan sets the limits the model must respect (decision 1). The hint order is used only as the
  fallback.

## 4. Automatic error type

**Why it matters:** the error profile, the "match the hint to the error" hint order and "what the AI
usually does next" all need sound error / wrong word / no answer. The model's training examples also
use these labels.

**Proposed:**
- A sound error is a word that sounds close to the expected one, compared with an offline
  pronunciation dictionary.
- A wrong word is a different real word.
- No answer is the "I'm not sure" button or a spoken "I don't know".
- Anything else is "unclassified".

**Known limit:** the speech recognizer tends to turn near-misses into real words ("tat" may come
back as "cat"). The error type is therefore approximate. Clinician corrections in review are the
safety net, and we report how often the scorer disagrees with them.

**Decision (Kuba, 2026-11-07): fully automatic, and the clinician never labels.**

Three separate parts:
1. **Speech recognition** (Whisper, pretrained): recording → transcript.
2. **Scorer** (fixed rules, no learning):
   - Correct / incorrect / unclear comes from matching the transcript against the expected word
     and its accepted alternatives.
   - The CMUdict pronunciation dictionary counts words that sound the same as a match ("pair" for
     "pear"), since speech can't tell them apart.
   - For an incorrect answer, the dictionary gives the error type: a sound error if the word sounds
     close to the target, a wrong word if it's a different word.
   - "I'm not sure" or "I don't know" counts as no answer. Anything else is "unclassified".
3. **Decision model** (DITTO): uses the score, the error type and the history to choose the next
   step.

The clinician listens to recordings and sees the transcripts and results, but never changes a
label. Their influence is only on the decision model: which hints, which pictures, when to move on.

**Known limit:** a misheard attempt stays in the summaries and in the context of training examples.
The safeguards are the "unclear" outcome (decision 6) and measuring the scorer on labeled test
recordings (decision 9).

**Implemented (2026-11-07):** `speech/scoring.py`, with tests in `tests/test_scoring.py`.
- A sound error is a phoneme edit distance of at most half the word's length (CMUdict).
- Words the dictionary doesn't know fall back to spelling distance. Spelling misses some
  near-misses, for example "kak" for "cat" comes out unclassified.
- `cmudict` (the Python package) is GPL-3.0. Fine for the hackathon; before distributing, replace it
  with the BSD-licensed dictionary file.
- Mac-voice check of 8 answers through Whisper `base.en`:
  - Whisper's confidence in isolated words is low even when it is right ("cat" 0.41, "dog" 0.17),
    so the unclear threshold is 0.1, not 0.45.
  - Whisper heard "umm, cat" as "I'm Kat", which is unclear, so the patient records again.
  - 7 of the 8 scored as intended.
  - Not a measurement: decision 9 does the real check.

**Open detail (for decision 10, Person 3):** the model chooses the *level* of the next picture; the
picture within that level is chosen automatically. Steering specific pictures ("more food words")
would need the model to choose among pictures too.

## 5. When an answer counts as correct

**Proposed:**
- Correct: the last meaningful word is the expected word or an accepted alternative. "a cat" and
  "cats" pass.
- Incorrect: a self-correction that ends on a wrong word ("cat… no, dog").
- Unclear: several naming words where the expected one isn't last.
- Never rewrite the transcript into the expected word.

**Decision (Kuba, 2026-11-07): single word only.**
- Only a one-word answer can be correct. Filler and article words are removed first: "a", "an",
  "the", "um", "uh", "er", "erm", "hmm".
- The word is correct if it is the expected word or an accepted alternative, including its plural
  ("cats"), or if it sounds exactly the same (CMUdict, for example "pair" for "pear").
- One other word is incorrect, and the error type comes from decision 4.
- Two or more words are unclear (decision 6), so "it's a cat" and "cat… no, dog" ask for another
  recording.
- The transcript is never rewritten.
- All 24 current item names and their alternatives are single words.

Fillers, however they are drawn out, are removed: "uuummm cat" is correct. Carrier phrases are not
removed: "it's a cat" has two or more words, so it is unclear and the patient records again
without it counting (Kuba, 2026-11-07).

## 6. Unclear recordings

**Proposed:**
- Silence, a very short recording, or low recognition confidence: ask for another recording, and
  don't count it.
- After 2 unclear recordings in a row: mark the picture "not assessed" and move on.

**Decision (Kuba, 2026-11-07): as proposed.**
- An unclear recording asks for another recording and doesn't count.
- After 2 unclear recordings in a row on the same picture, the picture is marked "not assessed" and
  practice moves to the next picture. It is neither a failure nor counted in the summaries, and the
  decision model isn't asked.
- Unclear means: silence, a recording that is too short, low recognition confidence, or two or more
  words (decision 5).

## 7. Hint audio

**Proposed:**
- Generate the audio offline with macOS text-to-speech and save the files.
- Add a "hear the whole word" clip, also played when a picture ends without a correct answer.
- First-sound clips are marked for a teammate to re-record, since text-to-speech says isolated
  sounds badly.

**Decision (Kuba, 2026-11-07): pre-recorded offline, simplest option.**
- A script generates 120 clips once with the macOS built-in voice (`say`): 24 pictures × 5 clips
  (first sound, first letter, sentence completion, meaning, whole word).
- The clips are saved as files next to the pictures.
- A clip is replaced by putting a recording with the same filename in its place; nothing else
  changes. First sounds are the best candidates.
- Hackathon only. If it is ever shared outside the team, check the license for macOS voices.

## 8. Storing recordings, and which speech model to use

**Proposed:**
- Keep recordings in git-ignored `runs/`, team voices only, with a delete command.
- Compare Whisper `base.en` with `small.en` on the test recordings (item 9), for accuracy and speed.

**Decision (Kuba, 2026-11-07):**
- Recordings are saved locally under git-ignored `runs/`, linked to their attempt, and never
  committed. Team voices only, no real patients. One command deletes them all.
- Speech recognition stays on Whisper `base.en`, about 0.5 s per answer. No model comparison.

## 9. Labeled test recordings

**Proposed:** each teammate records, for a subset of the pictures, the correct word, a wrong word,
a near-miss and silence (about 20 minutes each). The headline number is how often the scorer wrongly
says "correct". Team voices tell us nothing about aphasic speech, and we say so.

**Decision (Kuba, 2026-11-07): synthetic only.**
- A script generates labeled clips with several macOS voices: the correct word, an accepted
  alternative, a wrong word, a near-miss ("tat" for cat), two words, and silence or a very short
  clip.
- They run through the real speech model and scorer as an automated test, and the script writes an
  accuracy report.
- The headline number is how often the scorer wrongly says "correct".
- **Must be stated with the result:** synthetic speech is much cleaner than real or aphasic speech,
  so the accuracy is optimistic and says nothing about patients.

## 10. Data formats between the frontend, backend and model

**Proposed:**
- Write these first, as example JSON: an attempt, a score result, a decision, a clinician's
  training example, and a session summary.
- Session and attempt data follows the shapes in `product/mock/data.json`, so Person 1 can swap the
  mock out.

**Decision 10a (Kuba, 2026-11-07): contract document first.** Draft written:
[API_CONTRACT.md](API_CONTRACT.md), v0.1, waiting for review by Persons 1 and 3.
- Person 2 writes `docs/API_CONTRACT.md` before building. It lists each endpoint with example JSON.
- The shapes follow `product/mock/data.json`.
- Persons 1 and 3 review it before the backend is built.

**Decision 10b (Kuba, 2026-11-07): the model picks the level, and the picture follows from it.**
- The model's choice at each step is "retry with hint X" or "next picture at level N".
- The app then picks an unseen picture at that level.
- The clinician steers difficulty ("easier", "harder") by correcting those decisions or by writing
  notes, for example "after 2 missed pictures, go easier". DITTO learns from these.
- The model does not choose exact pictures. Clinician feedback is about level, so there would be
  nothing to teach it which picture to pick.


---

## Implemented (Person 2 hand-off, 2026-11-08)

- **1b note -> rule -> examples:** `service/steering.py`, `service/rule_drafter.py` (Claude API, structured outputs,
  `claude-opus-5-5` or `$CUEAGAIN_RULE_MODEL`; without `ANTHROPIC_API_KEY` the clinician fills the rule in by hand).
  The real-key check has not been run (no key).
- **1c / 8 training and activation:** `service/training.py`, `service/preview.py`. Patient adapters are `ditto:<run_id>`.
  Trained from the base model for now; the real run for `demo-ana` is deferred.
- **7 hint audio:** `scripts/make_hint_audio.py`, 120 clips in `frontend/hints/` (first sounds should be re-recorded by a person).
- **9 scorer accuracy:** `scripts/scorer_report.py`, `results/scorer_synthetic.md` (synthetic speech only, optimistic).
- **10 / 10.1 demo history:** `scripts/seed_demo_sessions.py`.
