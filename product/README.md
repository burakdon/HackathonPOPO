# CueAgain product prototype: patient tablet + clinician view

Open `product/index.html` in a browser (no server needed). Two views, switched in the top bar.

**Patient tablet**, modelled on a naming session with a speech therapist, with no helper assumed:
- Start screen with today's plan, then **How are you feeling today?** with four faces (no reading needed).
- One picture at a time. The person presses **Say it**; speech recognition scores the answer (in
  the CueAgain app; here a labelled demo strip stands in for it). **Hint** is always there.
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

## Server mode (real API)
Served by the app (`uv run --extra llm --extra speech uvicorn service.app:app --port 8000`), the page uses the
live API automatically: **http://localhost:8000/product/**. `?api=http://host:port` points it at another server;
`?mock` forces the mock data (also used when the file is opened directly, without a server).

In server mode only the server's patients are shown (simulated ones are labelled), and no mock data is used:
- **Patient tablet**: picture, scoring and every next step come from the server; "Say it" records the microphone;
  "Hint" sends "I'm not sure". "Microphone not working? Type the answer" is the fallback: the typed word is scored
  like a transcript (`POST /sessions/{sid}/typed`) and marked as typed in the review. `?api&fakemic` sends a
  silent clip instead of using the microphone. Not in the API yet: the end-of-session "try a few again" review;
  the mood check is shown but not sent.
- **Clinician**: session review with recordings, transcripts, scores and the model's decision after each attempt;
  "Prefer…" on a decision saves a correction as an example; a note becomes a rule (drafted by the server when it
  has an `ANTHROPIC_API_KEY`, otherwise filled in from the note's wording for the clinician to check) which, once
  confirmed, lists the matching moments to tick as examples; "Update <name>'s model" starts training with
  progress; "Show the result prepared earlier" takes a run trained before the demo on the same examples and says
  so on screen; the preview's three counts; activating the model; the patient's models and change log.

The demo-input strip (buttons standing in for speech recognition) was removed (docs/OPEN_DECISIONS.md, decision 2);
without a server the tablet asks for a typed answer instead.

After editing `template.html`, `uv run python -m scripts.render_product` rebuilds `index.html` and
`page.html` from the existing mock data without rerunning the simulator.
