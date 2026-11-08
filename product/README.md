# CueAgain product prototype: patient tablet + clinician view

Open `product/index.html` in a browser (no server needed). Two views, switched in the top bar.

**Patient tablet**, modelled on a naming session with a speech therapist, with no helper assumed:
- Start screen with today's plan, then **How are you feeling today?** with four faces (no reading needed).
- One picture at a time. The person presses **Say it**; speech recognition scores the answer (in
  the CueAgain app; here a labelled demo strip stands in for it). **Hint please** is always there.
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
