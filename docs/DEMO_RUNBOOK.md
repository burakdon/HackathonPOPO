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
- Patient tablet (real server, real microphone): **http://localhost:8000/product/?api** (pick Ana).
- The first decision of the day loads the model (about 4 s, plus about 3 s for the first answer). Start and end
  one warm-up session before the audience arrives, or run the rehearsal script on a copy (section 5).
- Close other heavy apps: model, Whisper and server use about 6 GB.

## 3. The demo, click by click

The clinician part goes through the API until the clinician view is wired (section 6). Set
`API=http://localhost:8000` and keep this file open to copy the commands. `jq` is optional.

**A. Ana practises (tablet).** Start a session. On the first picture give two wrong real words (for example
"table", then "spoon"; don't try to name "cat", Whisper often mishears it).
- After the first wrong word the model gives the **meaning hint** (a description).
- After the second wrong word it gives the **first-sound hint**. This is the moment the clinician will change.

**B. The clinician steers (API).**
```bash
# 1. the note (without ANTHROPIC_API_KEY it comes back with rule: null and asks for a hand-written rule)
curl -s -X POST $API/patients/demo-ana/notes -H 'content-type: application/json' \
  -d '{"text": "When meaning hints have repeatedly failed, try a first-letter hint before moving to another picture."}'
# -> note_id n-0001
# 2. confirm the rule
curl -s -X PUT $API/notes/n-0001/rule -H 'content-type: application/json' -d '{"rule": {
  "when": {"hints_tried_include": ["semantic"], "failed_attempts_on_picture": {"min": 2}},
  "then": {"op": "retry", "hint": "first_letter"},
  "summary": "After 2 or more failed tries on a picture where a meaning hint was already given -> give the first-letter hint."}}'
# 3. moments in Ana's past sessions that match (the model chose first_sound in every tickable one)
curl -s $API/notes/n-0001/matches | jq '.[] | select(.allowed) | {decision_id, word, model_chose, rule_says}'
# 4. tick the five decisions listed in runs/cueagain/prepared_demo.json ("tick_decision_ids")
curl -s -X POST $API/notes/n-0001/examples -H 'content-type: application/json' \
  -d "{\"decision_ids\": $(jq -c .tick_decision_ids runs/cueagain/prepared_demo.json)}"
# -> examples x-0001 ... x-0005
# 5. train: a real run starts (demo-ana-r2); show its progress once
curl -s -X POST $API/patients/demo-ana/training-runs
curl -s $API/training-runs/demo-ana-r2 | jq '{status, progress}'
# 6. say it plainly: "training takes about 7 minutes on this laptop; here is the same run prepared earlier"
curl -s -X POST $API/training-runs/demo-ana-r2/use-prepared -H 'content-type: application/json' \
  -d '{"prepared_run_id": "demo-ana-r1"}' | jq '{status, prepared, prepared_run_id, preview}'
# 7. activate Ana's adapter for her next sessions
curl -s -X POST $API/patients/demo-ana/active-model -H 'content-type: application/json' \
  -d '{"version": "ditto:demo-ana-r2"}' | jq .active_model
```
The preview to read out (real counts, `results/demo_prepared_run.md`):
- follows your examples: **5 of 5** (the current model: 0). These were trained on: it shows the update took.
- follows your rule in new situations: **5 of 10** (current: 0). Generated situations on other pictures.
- unchanged elsewhere: **9 of 10** held-out situations keep the same choice.

**C. Ana practises again (tablet).** New session, same two wrong words on the first picture: after the failed
meaning hint the model now gives the **first-letter hint**. Optional: show Ana's model list
(`curl -s $API/patients/demo-ana/models`) with "result prepared earlier".

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
The live run is cancelled by "use prepared" but its folder stays (`runs/ditto/demo-ana-r2`, manifest left as
"running"); if it is not removed, the next "train" fails with "runs/ditto/demo-ana-r2 already exists".

## 5. Rehearse without clicking

```bash
rm -rf runs/cueagain_e2e && cp -R runs/cueagain runs/cueagain_e2e
CUEAGAIN_DATA=runs/cueagain_e2e uv run --extra llm --extra speech uvicorn service.app:app --port 8000 &
uv run --extra llm --extra speech python -m scripts.demo_e2e_check --data runs/cueagain_e2e
# afterwards: stop the server, rm -rf runs/cueagain_e2e runs/ditto/demo-ana-r2*
```
Measured on an Apple M5 (24 GB): the hint after the failed meaning hint was first_sound before and first_letter
after; a recorded answer (Whisper + scoring + model) took 0.7-1.3 s; every clinician call (note, rule, matches,
ticks, start training, use prepared, activate) took 1-13 ms; the whole script 9 s.

## 6. Fallbacks

- **Microphone fails.** Use the tablet's "I'm not sure" button for misses (counted as no answer, so the first hint
  is the sentence hint, not the meaning hint), or run the rehearsal script live in a terminal: it speaks with a
  synthetic voice through the real endpoints and prints each hint. The new practice API has no typed-answer
  endpoint; the older developer console (http://localhost:8000/app/) accepts typed answers, labelled as manual
  entry, on its own practice flow.
- **The language model is unavailable** (not downloaded, out of memory): every decision falls back to the plan's
  hint order and says `source: fallback` with the reason; the session keeps working.
- **Plan B: DITTO on the RL policy.** The PPO therapist learns from a few clinician overrides in under a second
  (`POST /learn_from_overrides`); steps in the README's "Demo script" section (needs `agent: ppo` in
  `configs/service.yaml`, the default).
