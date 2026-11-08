# Contract sample responses

These are sample responses for every endpoint in [../API_CONTRACT.md](../API_CONTRACT.md), for
building the frontend before the backend exists.

They are **generated, not hand-written**. `scripts/build_contract_examples.py` builds every file
through the same response models the real endpoints will return (`core/contract.py`). A test
(`tests/test_contract_examples.py`) fails if a file is out of date or no longer matches its model.
Don't edit these files by hand; change the script and run:

```bash
uv run python -m scripts.build_contract_examples
```

## Use them from the frontend: the mock server

```bash
uv run uvicorn service.contract_mock:app --port 8001
```

- It serves every sample at its real path, for example `GET http://localhost:8001/pictures`.
  Any ID in a path works.
- CORS is open, so `product/index.html` can call it straight from a `file://` page.
- **Practice is scripted.** `POST /patients/{id}/sessions` starts (or rewinds) one 6-picture
  session. Each `POST /sessions/{id}/recordings` or `/no-answer` then returns the next scripted
  step, whatever audio you send. The `X-Mock-Step` header says which step it is, and
  `X-Mock-Endpoint-Matches` says whether the script expected "recording" or "I'm not sure" at
  that point.
- Images are real (`/media/images/...`). Hint clips and patient recordings are a **placeholder
  tone** until the real clips exist.
- Switching to the real backend is a change of base URL. The paths and shapes stay the same.

## The scripted session (demo patient "Ana", simulated)

| # | picture | what happens | result |
|---|---|---|---|
| 1 | cat | "cat" | independent |
| 2 | dog | "bone" (wrong word) → meaning hint → "uuummm dog" | assisted |
| 3 | apple | "it's an apple" (unclear, not counted) → "I'm not sure" → sentence hint → "apple" | assisted |
| 4 | house | silence, silence (2 unclear) | not assessed |
| 5 | car | "bus" → meaning hint → "truck" → sentence hint → "train" → show the word | unsuccessful |
| 6 | tree | "free" (sound error) → first-sound hint → "tree" | assisted |

Picture 5 is the demo situation. After two failed tries with a meaning hint already given, the
model chose the sentence hint (decision `s-demo-01-d07`). The clinician's note asks for the first-sound
hint there. `steering/GET_matches.json` shows `s-demo-01-d07` as a match that can be ticked, and `s-demo-01-d08`
as a match the rule can't apply to, because no tries were left.

## Folders

| folder | contract section |
|---|---|
| `pictures/` | §2 pictures and hint audio |
| `patients/` | §3 patients, plans, change log |
| `practice/` | §4 session start, the 13 scripted steps in `script/`, session state, end, error examples |
| `review/` | §6 session rows and the full session review |
| `steering/` | §7 a correction, note → rule (and a note that gives no rule), confirmed rule, matches, examples |
| `training/` | §8 training run (queued, running, completed), the prepared result, model list, activation |

- `*.request.json` files are example request bodies.
- `index.json` lists every file with its method, path, status code and model.

## Not real data

- Scores and the model's choices are scripted. Neither the scorer nor the model was called.
- The preview numbers in `training/` are **illustrative placeholders, not measurements**.
