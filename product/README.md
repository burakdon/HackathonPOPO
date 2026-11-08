# CueAgain product prototype: patient tablet + clinician view

Open `product/index.html` in a browser (no server needed). Two views, switched in the top bar:

- **Patient tablet:** start screen with today's plan, one picture at a time, the helper taps what
  happened (speech recognition runs in the CueAgain app), the smallest helpful hint first and
  stronger ones on retry (clinical cueing hierarchy), show-and-repeat after the last try, difficulty
  that follows first-try success, a break after repeated misses, and an encouraging end summary.
  A finished session appears in the clinician view as "Today (live)".
- **Clinician:** caseload with weekly practice minutes and flags, patient page (naming trend, error
  profile, what the AI does after each error type), every session's decisions with the AI's reason
  and an "I'd do this differently" control, typed feedback that becomes proposed plan changes she
  approves, plan settings, a change log, and **Teach the AI**.

## Where the data comes from
`product/mock/data.json` is generated, not typed: `uv run python -m scripts.build_product_mock`
runs four patients for two weeks in the data-grounded simulator (`configs/sim_pnt.yaml`) with the
trained agent (`models/ppo_pnt_seed0.zip`). Maria and James are the personas in `docs/PERSONAS.md`.
James's **Teach the AI** result is computed by the script for real (DITTO-style tuning of the trained
agent on 8 simulated corrections of the same kind; adoption measured on 160 held-out decisions) and
shown when the clinician presses the button. Other patients' corrections are saved, not trained live.

The script then builds `index.html` (and `page.html`, the same page without the document wrapper)
from `template.html` with the data and the 24 pictures of `data/items.json` inlined. **Edit
`template.html`, then rerun the script.**

## Moving to the real back end
All data access is in `template.html`: `MOCK.patients` (caseload, sessions, plans), `ITEMS` and
`PICS` (pictures). To go live, replace them with calls to the service (practice sessions,
`/override`, `/learn_from_overrides`, a plan endpoint) that return the same shapes as
`mock/data.json`. Feedback-to-plan suggestions use simple rules here (`parseFeedback`); the product
would have a language model draft them, with the clinician approving each change.
