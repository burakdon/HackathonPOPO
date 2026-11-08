# DITTO demo runbook

The demo: Ana (a simulated patient) practises; the clinician writes a note, confirms it as a rule, ticks matching
moments from Ana's past sessions and starts training; the result shown was **prepared earlier** (and we say so);
after activating it, Ana's next session gives the clinician's preferred hint. Rehearsed end to end on an Apple M5
laptop through the real server (`scripts/demo_e2e_check.py`, results at the end).

The demo rule is **"after a failed meaning hint, give the first-letter hint"**. The shared adapter already gives the
first-sound hint there (in 11 of 11 seeded situations), so first-letter makes the change visible.

## 1. Set up the demo laptop (once)

```bash
git clone -b feature/ditto-training https://github.com/burakdon/HackathonPOPO.git && cd HackathonPOPO   # or dev after the merge
uv sync --extra llm --extra speech                        # Python 3.11; Qwen3 + LoRA, Whisper, CMUdict
uv run --extra llm --extra speech python -m scripts.ditto_golden_check   # downloads Qwen3-1.7B (~4 GB) once; must print "golden test passed"
```

Then get the demo data, either copied from the machine that prepared it or made again (about 10 minutes).

**Copy** (these are git-ignored, so they are not in the clone):

| path | what it is |
|---|---|
| `runs/cueagain/patients/demo-ana.json` | Ana, plan, active model `ditto:sft200-g2-s0` |
| `runs/cueagain/sessions/` and `runs/cueagain/audio/` | the 3 seeded sessions (real Whisper transcripts, real model decisions, playable recordings) |
| `runs/cueagain/seed_demo_sessions.json` | which sessions are seeded (for `--reset`) |
| `runs/cueagain/training/demo-ana-r1.json` and `demo-ana-r1.examples.jsonl` | the prepared run's record (status completed, preview) |
| `runs/cueagain/prepared_demo.json` | what to tick in the demo and the prepared run id |
| `runs/ditto/demo-ana-r1/` | the prepared run's adapter and manifest |

```bash
# on the preparing machine (before any rehearsal changes the data)
tar czf demo_runs.tgz runs/cueagain runs/ditto/demo-ana-r1
# on the demo laptop, in the repo
tar xzf demo_runs.tgz && cp -R runs/cueagain runs/cueagain.backup      # keep a clean copy for resets
```
Copy only `runs/ditto/demo-ana-r1`: every completed run in `runs/ditto/` shows up in the patient's model list.
`runs/cueagain` must contain no `notes/` or `examples/` yet (the demo creates them; the prepared run expects the
example IDs x-0001 to x-0005).

**Or make it again** on the demo laptop (same results up to run-to-run variation):
```bash
uv run --extra llm --extra speech python -m scripts.seed_demo_sessions --reset   # ~3 min: 3 sessions for demo-ana
uv run --extra llm --extra speech python -m scripts.prepare_demo_run             # ~7 min: real training + preview
cp -R runs/cueagain runs/cueagain.backup
```

## 2. Start everything (demo day)

```bash
uv run --extra llm --extra speech uvicorn service.app:app --host 0.0.0.0 --port 8000
curl -s localhost:8000/health                              # {"status":"ok",...}
```
- Open **http://localhost:8000/product/**. Served by the app, the page uses the live API and shows only the
  server's patients (Ana, labelled "Simulated"). Top bar: **Patient tablet** and **Clinician**.
- The first decision of the day loads the model (about 4 s, plus about 3 s for the first answer). Do one warm-up
  answer before the audience arrives (then reset, section 4), or run the rehearsal script on a copy (section 5).
- Close other heavy apps: model, Whisper and server use about 6 GB.

## 3. The demo, click by click (product/)

**A. Ana practises (Patient tablet).**
1. Patient tablet → patient "Ana (demo) · server" → **Start** → pick a face.
2. The first picture is the **dog** (sessions no longer start with "cat", which Whisper often mishears).
3. **Say it** and give a wrong real word ("table"), press **Done**: the model gives the **meaning hint** ("A pet
   that wags its tail and fetches sticks.").
4. **Try again** with another wrong word ("spoon"): the model gives the **first-sound hint** ("It starts with the
   sound d…"). This is the moment the clinician will change.
   If the microphone fails: **Microphone not working? Type the answer** (typed answers are marked as typed).

**B. The clinician steers (Clinician).**
1. Clinician → **Ana (demo)**. The session review shows every picture, what Ana said (▶ plays the recording),
   the score and why, and **the model's decision after each attempt**. Open the latest session and point at the
   dog: after the failed meaning hint the model chose the first sound. ("Prefer…" next to any decision saves a
   correction as an example; not needed for the demo, and it would change the example count, see below.)
2. **Teach the model**: type (or keep the placeholder) "When meaning hints have repeatedly failed, try a
   first-letter hint before moving to another picture." → **Turn the note into a rule**. Without an
   `ANTHROPIC_API_KEY` the screen says so and fills the rule in from the note's wording: *meaning hint already given*,
   *at least 2 failed tries*, *give the first-letter hint*. Check it → **Confirm the rule**.
3. The matching moments from Ana's past sessions appear (the model chose the first sound in every one that can be
   used). Tick the five in `runs/cueagain/prepared_demo.json` ("tick_decision_ids"; on the prepared data: dog and
   house in the first session, apple, car and tree in the second) → **Use 5 ticked moments as examples**. They
   appear under **Examples for Ana** as "Not trained yet" (x-0001 to x-0005).
4. **Update Ana's model**: training really starts; the progress bar shows "Learning from the examples", then the
   DITTO rounds. Say: "on this laptop this takes about 7 minutes; here is the same run, prepared earlier on the same
   examples" → **Show the result prepared earlier**.
5. The screen says **Prepared earlier** (run demo-ana-r1, same 5 examples, live training stopped) and shows the
   three counts (real, `results/demo_prepared_run.md`):
   - follows your examples **5 of 5** (current model 0 of 5): trained on these, it shows the update took;
   - follows your rule in new situations **5 of 10** (current 0 of 10): generated situations on other pictures;
   - unchanged elsewhere **9 of 10** held-out situations.
6. **Use this model for Ana's next sessions** → "In use for Ana's next sessions"; the header and **Ana's models**
   now show `ditto:demo-ana-r2`, "result prepared earlier (demo-ana-r1)".

**C. Ana practises again (Patient tablet).** Start a new session, same two wrong words on the dog: after the failed
meaning hint the model now gives the **first-letter hint** ("It starts with the letter D.").

"Use prepared" accepts the prepared run only when both runs were trained on the same example IDs (x-0001 to
x-0005), so make no corrections or other examples before step B4.

**Backup: the same clinician steps through the API** (if the page misbehaves; `API=http://localhost:8000`):
```bash
curl -s -X POST $API/patients/demo-ana/notes -H 'content-type: application/json' \
  -d '{"text": "When meaning hints have repeatedly failed, try a first-letter hint before moving to another picture."}'
curl -s -X PUT $API/notes/n-0001/rule -H 'content-type: application/json' -d '{"rule": {
  "when": {"hints_tried_include": ["semantic"], "failed_attempts_on_picture": {"min": 2}},
  "then": {"op": "retry", "hint": "first_letter"},
  "summary": "After 2 or more failed tries on a picture where a meaning hint was already given -> give the first-letter hint."}}'
curl -s -X POST $API/notes/n-0001/examples -H 'content-type: application/json' \
  -d "{\"decision_ids\": $(jq -c .tick_decision_ids runs/cueagain/prepared_demo.json)}"
curl -s -X POST $API/patients/demo-ana/training-runs
curl -s -X POST $API/training-runs/demo-ana-r2/use-prepared -H 'content-type: application/json' -d '{"prepared_run_id": "demo-ana-r1"}'
curl -s -X POST $API/patients/demo-ana/active-model -H 'content-type: application/json' -d '{"version": "ditto:demo-ana-r2"}'
```

What to say: the model is a 1.7B language model with a small adapter; it agrees with the demonstrator on 75% of
55 situations it never saw (the untrained model 29%); the clinician's update changed this one decision for Ana
only, and left most other decisions as they were. It is a practice-tool prototype on fictional preferences, not a
clinical result.

## 4. Reset between rehearsals

```bash
# stop the server first (Ctrl+C), then
pkill -f "training.patient_ditto" ; rm -rf runs/ditto/demo-ana-r2 runs/ditto/demo-ana-r2.config.yaml
rm -rf runs/cueagain && cp -R runs/cueagain.backup runs/cueagain
```
"Use prepared" stops the live run and marks its manifest "cancelled", but its folder stays
(`runs/ditto/demo-ana-r2`); after restoring the data, the next "train" reuses that run id and fails with
"runs/ditto/demo-ana-r2 already exists" unless the folder is removed.

## 5. Rehearse without clicking

```bash
rm -rf runs/cueagain_e2e && cp -R runs/cueagain runs/cueagain_e2e
CUEAGAIN_DATA=runs/cueagain_e2e uv run --extra llm --extra speech uvicorn service.app:app --port 8000 &
uv run --extra llm --extra speech python -m scripts.demo_e2e_check --data runs/cueagain_e2e
# afterwards: stop the server, rm -rf runs/cueagain_e2e runs/ditto/demo-ana-r2*
```
Measured on an Apple M5 (24 GB): the hint after the failed meaning hint was first_sound before and first_letter
after; a recorded answer (Whisper + scoring + model) took 0.7-1.6 s; every clinician call (note, rule, matches,
ticks, start training, use prepared, activate) took 1-13 ms; the whole script 10 s. The same demo was clicked
through in product/ with a headless Chrome (typed answers instead of the microphone): every step above worked, a
typed answer took 0.7-1.1 s once the model was loaded, each clinician click 50-350 ms.

## 6. Fallbacks

- **Microphone fails.** On the tablet, **Microphone not working? Type the answer**: the typed word is scored like
  a transcript (`POST /sessions/{sid}/typed`) and marked "Typed answer (microphone fallback)" in the review; the
  demo works the same with typed "table" and "spoon". The rehearsal script (section 5) is a second fallback.
- **The language model is unavailable** (not downloaded, out of memory): every decision falls back to the plan's
  hint order and says `source: fallback` with the reason; the session keeps working.
- **Plan B: DITTO on the RL policy.** The PPO therapist learns from a few clinician overrides in under a second
  (`POST /learn_from_overrides`); steps in the README's "Demo script" section (needs `agent: ppo` in
  `configs/service.yaml`, the default).
