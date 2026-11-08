# NeuroCue product prototype: patient tablet + clinician view

Open `product/index.html` in a browser (no server needed). Two views, switched in the top bar.

**Patient tablet**, modelled on a naming session with a speech therapist, with no helper assumed:
- Start screen with today's plan, then **How are you feeling today?** with four faces (no reading needed).
- One picture at a time. The person presses **Say it**; speech recognition scores the answer (in
  the NeuroCue app; here a labelled demo strip stands in for it). **Hint** is always there.
- After a miss, the smallest helpful hint for that kind of mistake, then stronger ones (cueing
  hierarchy). With "meaning first", a wrong word starts with describing the object: group, use,
  where you find it (semantic feature analysis). Hints can be read aloud.
- After a hint works: say it once more. After the last try: hear it and say it together.
- Difficulty follows first-try success; a break is offered after repeated misses.
- The session ends by trying the pictures that needed help again, without hints, then a short,
  encouraging summary. The session appears in the clinician view as "Today".

**Clinician**, in plain clinical language, larger text, no mention of models:
- Your patients: last practice, minutes against the weekly goal, one status (needs attention,
  improving, on track).
- Patient page: three numbers (named on their own, practice this week, flags), the naming chart with
  mood before each session underneath, a flag when mood is low before 3 of the last 4 sessions
  (suggesting depression screening such as SADQ-10, or referral), "What we're seeing" (main type of miss, hardest words, words now reliable,
  stopping early, review recall) and "Which hints help" (success after each hint type).
- Session review: the pictures that needed help, with "Suggest a different hint" for next time.
  Repeated suggestions of the same kind offer to become the patient's standing approach.
- Notes and suggestions: a typed note becomes proposed updates she approves.
- Exercise plan: practice days, minutes, pictures, level, hint approach, tries, breaks, when to
  move up, personal words. Plan history records every change.

Behind the scenes, a clinician's suggestions are the demonstrations the decision model learns from
(DITTO). The clinician never sees that; she sees her patient's next sessions follow her approach.

## Where the data comes from
`product/mock/data.json` is generated, not typed: `uv run python -m scripts.build_product_mock`
runs four patients for two weeks in the data-grounded simulator (`configs/sim_pnt.yaml`) with the
trained agent (`models/ppo_pnt_seed0.zip`). Maria and James are the personas in `docs/PERSONAS.md`.
James's standing-approach result (the sessions after the clinician's instruction) is computed by the script for real (DITTO-style tuning of the trained
agent on 8 simulated corrections of the same kind; adoption measured on 160 held-out decisions) and
those sessions appear when she applies the instruction. For other patients the instruction updates the hint approach in the plan.

The script then builds `index.html` (and `page.html`, the same page without the document wrapper)
from `template.html` with the data and the 24 pictures of `data/items.json` inlined. **Edit
`template.html`, then rerun the script.**

## Moving to the real back end
All data access is in `template.html`: `MOCK.patients` (caseload, sessions, plans), `ITEMS` and
`PICS` (pictures). To go live, replace them with calls to the service (practice sessions,
`/override`, `/learn_from_overrides`, a plan endpoint) that return the same shapes as
`mock/data.json`. Feedback-to-plan suggestions use simple rules here (`parseFeedback`); the product
would have a language model draft them, with the clinician approving each change.

## Server mode (patient tablet, real API)
Open the page with `?api` to run the tablet against the server API (`docs/API_CONTRACT.md` §4):

```bash
uv run uvicorn service.contract_mock:app --port 8001
```
then `http://localhost:8001/product/index.html?api#patient`. Until the backend exists, the mock server
plays a scripted session for the demo patient Ana (`docs/contract_examples/`). Pointing `?api=http://host:port`
at the real backend later needs no page changes.

- Server patients appear first in the tablet's patient list (marked "server"). For them the picture,
  scoring and every next step come from the server, "Say it" records the microphone, and "Hint"
  sends "I'm not sure". The demo-input strip is hidden.
- Without `?api` the page makes no requests and works exactly as before.
- Not in the API yet, so skipped in server mode: the end-of-session "try a few again" review. The
  mood check is shown but not sent.
- `?api&fakemic` sends a short silent clip instead of using the microphone (for browsers that block it).

After editing `template.html`, `uv run python -m scripts.render_product` rebuilds `index.html` and
`page.html` from the existing mock data without rerunning the simulator.

## Server mode: clinician view (Ana)
With `?api`, server patients (the demo patient Ana) get a clinician page built from the server
(`docs/API_CONTRACT.md` §3, §6, §7, §8). Maria, James, Ruth and David stay as hardcoded simulator data,
labelled "Simulated patient", display only (charts and session list, no corrections, notes or plan
changes), so nobody compares their precomputed results with Ana's real ones.

Ana's page, top to bottom:
- Header and numbers: `GET /patients/{id}` (with the "Simulated" badge when flagged), last session,
  pending corrections.
- Naming per session: stacked bars (independent / with hint / unsuccessful) from
  `GET /patients/{id}/sessions`, plus the error mix.
- Session review: `GET /sessions/{id}/review`. Each picture with its tries (transcript, result,
  `meaning` shown as "Wrong word", the hint shown, the scorer's reason, a play button for `audio_url`),
  and after each try the next step the model chose and its `source` ("fallback (reason)" when
  `fallback_reason` is set).
- "I'd do this differently…" on each decision: that decision's other `options` →
  `POST /decisions/{id}/correction` with an optional note.
- Clinician notes: `POST /patients/{id}/notes` → the drafted rule → Confirm (`PUT /notes/{id}/rule`).
  If `rule` is `null`, the `problem` is shown with a "when ___ → then ___" form that builds the rule.
  Then `GET /notes/{id}/matches` lists past situations with checkboxes (rows with `allowed: false` or
  `already_agrees` are disabled) → `POST /notes/{id}/examples`.
- Corrections: `GET /patients/{id}/examples`, pending and already learned shown separately.
- **Train (demo shortcut):** shows progress for about 10 seconds, then activates the prepared model with
  `POST /patients/{id}/active-model` and says "Trained earlier on these examples (training takes about
  5 minutes)". It never calls `POST /training-runs` (a real 5-minute GPU job).
  The prepared model ID defaults to `ditto:demo-ana-after`; override with `?prepared=<id>`, and the
  duration with `?trainsec=<seconds>`.
- Model in use: `GET /patients/{id}/models` + the same activate call, to reset Ana to the "before"
  model between takes.
- Change log: `GET /patients/{id}/log`.

Opening the clinician view or picking Ana reloads her record, so a session just done on the tablet
shows up straight away.
