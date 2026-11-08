# DITTO model handoff: integrating the delivered model without merging

For integrating the delivered decision model into another branch (here `Kuba_branch`) **without merging
`feature/ditto-training`**, while keeping its decisions exactly as trained. Everything below refers to the
code on `feature/ditto-training`. If your port passes the golden test (section 6), it decides like the
reference.

## 1. What the model is

| | |
|---|---|
| base model | `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (pinned) |
| adapter | LoRA in `models/ditto_llm/sft200-g2-s0/` (r 16, alpha 32, q/k/v/o projections, 25.7 MB) |
| model version id | `ditto:sft200-g2-s0` |
| how it decides | **option scoring**: the server passes the allowed actions; the model computes the log-probability of each action's JSON and the most likely one is taken. There is no free text generation, so the output is always one of the options |
| training | SFT 200 steps on 161 demonstrations, then gentle DITTO (3 rounds, learning rate 2e-5, beta 0.2, RPO 0.5); the delivered stage (round 3) was chosen on the 33-situation dev split before any test evaluation |

Results on **55 situations never used in training** (`results/ditto_colab_r2_test.md`, 3 seeds, 95%
bootstrap intervals):

| model | agrees with the demonstrator |
|---|---|
| untrained Qwen3-1.7B | 29% (CI 18-42%) |
| Qwen3-1.7B with 20 demonstrations in the prompt | 38% (CI 25-51%) |
| SFT only | 77% (CI 68-85%) |
| **delivered DITTO adapter** | **75% (CI 65-84%)** |

Latency per decision: about 80 ms on an A100; on an Apple M5 laptop (MPS) 0.2-0.5 s with 2-4 options and
about 1 s with 7. Loading base model plus adapter takes about 4 s once the model is in the Hugging Face cache
(the first download is about 4 GB).

## 2. Files to read and files to copy

The inference files below are **unchanged on `Kuba_branch`** relative to the base `feature/ditto-training`
started from (`Sabri_Branch`), so taking my version of each replaces nothing of yours. Check before copying:
`git diff origin/Sabri_Branch origin/Kuba_branch -- <file>` should print nothing.

| file | what it does for inference |
|---|---|
| `models/ditto_llm/sft200-g2-s0/` | the adapter (`adapter_model.safetensors`, `adapter_config.json`) and its `manifest.json` (base model, revision, `prompt_format: v2`, `decoding: score`, `delivered_stage: round3`) |
| `core/practice.py` | the prompt: `render_context(..., fmt="v2")`, `option_label`, `HINT_WORDS`, `TASK_INSTRUCTION_V2`, `messages()`; the action format (`format_action`, `parse_action`); `contract_options()`; `LLMConfig` fields `decoding`, `prompt_format`, `models_dir` |
| `agents/llm_runtime.py` | loads the base model once and LoRA adapters on top; chat template settings (`TEMPLATE_SETTINGS`); `score()` = option scoring (batched, or one option at a time for long prompts) |
| `training/ditto.py` | `Encoder` and `answer_logps`, which `score()` imports so that inference scores exactly what training optimised (also the trainer itself) |
| `agents/llm_policy.py` | `LLMDecisionPolicy`: renders the prompt, runs `score()` with a timeout, validates the choice, falls back |
| `agents/registry.py` | maps a version (`ditto:<run_id>`) to its adapter folder in `runs/ditto/` or `models/ditto_llm/`; refuses an adapter whose base model, revision or prompt format differs from the config |
| `configs/practice.yaml` | serving settings: base model and revision, `dtype: bfloat16`, `decoding: score`, `prompt_format: v2`, `models_dir`, `default_model: ditto:sft200-g2-s0` (yours is unchanged from the base, so it can be replaced) |
| `training/team_format.py` (optional) | `patient_state.json` <-> `DecisionContext`; `decide_action_id()` returns `{"action_id"}` |
| `scripts/ditto_golden_check.py`, `docs/ditto_golden_test.json` | the golden test (section 6) |

Copy the adapter folder without merging:

```bash
git fetch origin
git checkout origin/feature/ditto-training -- models/ditto_llm/sft200-g2-s0
```

And the inference code, the config and the golden test, the same way:

```bash
git checkout origin/feature/ditto-training -- core/practice.py agents/llm_runtime.py agents/llm_policy.py \
    agents/registry.py training/ditto.py configs/practice.yaml \
    scripts/ditto_golden_check.py docs/ditto_golden_test.json
```

Keep your own `pyproject.toml` (it differs from mine by 2 lines); the `llm` extra pins are the same on both
branches: `accelerate==1.15.0`, `peft==0.21.2`, `transformers==5.22.2` (torch 2.15.0 was used). On Colab,
uninstall the preinstalled `torchao` first: peft refuses torchao 0.10.

## 3. Invariants: change any of these and the model behaves differently

**Base model, revision and dtype** (`configs/practice.yaml`). The adapter was trained on exactly this base,
in bfloat16:
```yaml
llm:
  base_model: Qwen/Qwen3-1.7B
  revision: 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e
  dtype: bfloat16
  decoding: score
  prompt_format: v2
history_in_prompt: 4
```

**Chat template**: Qwen3's own template, with a generation prompt and thinking **disabled** (the template
then inserts an empty `<think></think>` block before the answer) (`agents/llm_runtime.py`):
```python
TEMPLATE_SETTINGS = {"add_generation_prompt": True, "enable_thinking": False}

def chat_prompt(tokenizer, msgs: list[dict[str, str]]) -> str:
    return tokenizer.apply_chat_template(msgs, tokenize=False, **TEMPLATE_SETTINGS)
```

**System prompt**, word for word (`core/practice.py`, `TASK_INSTRUCTION_V2`):
```python
TASK_INSTRUCTION_V2 = (
    "You help run picture-naming practice for an adult practising word finding. After each "
    "confirmed attempt you choose what happens next: retry the same picture (optionally with one "
    "of its predefined hints), or move on to a new picture at a difficulty level (tier 1 easiest, "
    "3 hardest). Each option is a JSON object followed by its meaning in words. Choose exactly one "
    "option and reply with its JSON object exactly as written, with nothing else."
)
```
Messages are `[system: TASK_INSTRUCTION_V2, user: render_context(...)]`: no few-shot examples, no notes,
no patient id.

**How the context is rendered** (`core/practice.py`). Fields used: the current picture's word, tier,
attempt count and hints shown; the latest attempt; **at most the 4 attempts before it** (`history_in_prompt:
4`, oldest first); session minutes. Response time has one decimal, session minutes none. The exercise id is
never shown to the model.
```python
HINT_WORDS = {"none": "no hint", "first_sound": "first-sound hint", "first_letter": "first-letter hint",
              "sentence_frame": "sentence-completion hint", "semantic": "meaning hint (a description)"}

def _attempt_line(a: AttemptSummary, fmt: str = "v1") -> str:
    hint = HINT_WORDS[a.hint].split(" (")[0] if fmt == "v2" else ("no hint" if a.hint == "none" else f"hint {a.hint}")
    if a.correct:
        res = "correct"
    elif a.error == "unclassified":
        res = "incorrect (type not identified)"
    elif a.error == "no_response":
        res = "incorrect (no response)"
    else:
        res = f"incorrect ({a.error} error)"
    return f"{a.word}, tier {a.tier}, {hint}: {res}, {a.response_time_s:.1f} s"

def render_context(ctx, options, history_in_prompt, fmt="v1") -> str:
    cur = ctx.current
    earlier = ctx.history[:-1][-history_in_prompt:] if history_in_prompt else []
    if fmt == "v2":
        shown = ", ".join(HINT_WORDS[h].split(" (")[0] for h in cur.hints_shown) if cur.hints_shown else "none"
        opts = [f"{format_action(o)} = {option_label(o, cur.tier)}" for o in options]
    ...
    lines = [
        f"Current picture: {cur.word} (tier {cur.tier})",
        f"Attempts on this picture: {cur.attempts}; hints already shown for it: {shown}",
        f"Latest attempt: {_attempt_line(ctx.latest, fmt)}",
        "Earlier attempts this session (oldest first):" if earlier else "Earlier attempts this session: none",
        *[f"- {_attempt_line(a, fmt)}" for a in earlier],
        f"Session time: {ctx.session_minutes:.0f} min",
        "Options:",
        *opts,
    ]
    return "\n".join(lines)
```

**How options are rendered**: the action JSON, then ` = ` and a label with the level relative to the
current picture, in the order the server passes them:
```python
def option_label(a: PracticeAction, current_tier: int) -> str:
    if a.operation == "retry_current":
        return f"same picture again with the {HINT_WORDS[a.cue]}"
    rel = {0: "same level", -1: "one level easier", 1: "one level harder"}.get(a.tier - current_tier, f"tier {a.tier}")
    return f"new picture, {rel} (tier {a.tier})"
```

**Action format**: compact JSON, exactly these keys and no spaces (`format_action`):
`{"op":"retry","hint":"first_sound"}` and `{"op":"next","tier":2}`. Hints are `first_sound`,
`first_letter`, `sentence_frame` and `semantic`; tiers are 1-3.

**How options are scored** (`agents/llm_runtime.py` `score()`, using `training/ditto.py`): each option's
answer is its JSON followed by `<|im_end|>`; its score is the **sum** of the token log-probabilities of
those answer tokens given the full chat prompt. The option with the highest score is chosen (the first on
an exact tie). Prompts are left-padded in a batch, with explicit position ids; this gives the same scores as
scoring one option at a time.
```python
def answer_ids(self, text: str) -> list[int]:
    ids = self.tok(text, add_special_tokens=False)["input_ids"]
    return ids if ids and ids[-1] == self.end else ids + [self.end]      # self.end = <|im_end|>

lps = [float(x) for x in answer_logps(self.model, enc, [(prompt, enc.answer_ids(c)) for c in candidates])]
```

The exact text the model sees for golden case g2, produced by the code:
```text
<|im_start|>system
You help run picture-naming practice for an adult practising word finding. After each confirmed attempt you choose what happens next: retry the same picture (optionally with one of its predefined hints), or move on to a new picture at a difficulty level (tier 1 easiest, 3 hardest). Each option is a JSON object followed by its meaning in words. Choose exactly one option and reply with its JSON object exactly as written, with nothing else.<|im_end|>
<|im_start|>user
Current picture: glasses (tier 2)
Attempts on this picture: 2; hints already shown for it: meaning hint
Latest attempt: glasses, tier 2, meaning hint: incorrect (meaning error), 6.2 s
Earlier attempts this session (oldest first):
- dog, tier 1, no hint: correct, 3.1 s
- glasses, tier 2, no hint: incorrect (meaning error), 5.0 s
Session time: 9 min
Options:
{"op":"retry","hint":"first_sound"} = same picture again with the first-sound hint
{"op":"retry","hint":"first_letter"} = same picture again with the first-letter hint
{"op":"retry","hint":"sentence_frame"} = same picture again with the sentence-completion hint<|im_end|>
<|im_start|>assistant
<think>

</think>

{"op":"retry","hint":"first_sound"}<|im_end|>
```
(the last line is the scored answer for the first option; each option is scored in the same position).

## 4. Required input fields

The model reads a `DecisionContext` (`core/practice.py`, contract section 5):

| field | used for | if missing or wrong |
|---|---|---|
| `current.word`, `current.tier` | first line; levels in option labels are relative to `tier` | required (validation error) |
| `current.attempts` | counted attempts on this picture | defaults to 0, which misdescribes the situation: always send it |
| `current.hints_shown` | hints already shown on this picture, **including the hint visible during the latest attempt** | defaults to empty: the model then thinks no hint was tried |
| `history[]` | oldest first; the **last entry must be the latest attempt on the current picture**; only the last 5 entries (latest + 4) reach the prompt | at least one entry required; latest on another picture is a validation error |
| `history[].hint` | hint visible during that attempt (`none` if independent) | required |
| `history[].correct`, `history[].error` | `error` is `sound` / `meaning` / `no_response` / `unclassified`, and `null` iff correct; unclear recordings are never included | inconsistent pair is a validation error |
| `history[].response_time_s` | the demonstrator moves to a harder level after a fast (at most 4 s) independent correct answer | required by validation; if your data has none, send 5.0 (what `training/team_format.py` uses), which never triggers "harder" |
| `session_minutes` | the demonstrator moves to an easier picture after an error at 20 minutes or more | required by validation; 0 if unknown (rule G then never applies) |
| `history[].exercise_id` | only to check that the latest attempt is on the current picture; never shown | required |

The **options** are part of the input: the model only ranks what it is given. Build them on the server
(contract section 5 and decision 1d): no hint already shown on the picture, no plain retry, and after a wrong
answer with tries left hints only; after a correct answer or with no tries left, next-level options within
one level of the current one.

## 5. Reference `decide()` (contract section 5)

Built on the copied files, so the prompt and scoring are the reference ones. Patient adapters trained with
`training/ditto.py` live in `runs/ditto/<run_id>/` with a manifest and load as `ditto:<run_id>`; delivered
ones live in `models/ditto_llm/<run_id>/`. The fallback is the plan's fixed hint order (decision 1).

```python
import concurrent.futures, json, time
from agents.registry import ModelRegistry
from core.practice import DecisionContext, PracticeAction, format_action, load_practice

PCFG = load_practice()                    # decoding: score, prompt_format: v2, default ditto:sft200-g2-s0
REGISTRY = ModelRegistry(PCFG)            # base model loaded once; adapters added on first use
ACTIVE: dict[str, str] = {}               # patient_id -> version, set by POST /patients/{id}/active-model


def fallback_action(ctx: DecisionContext, options: list[PracticeAction], hint_order: list[str]) -> PracticeAction:
    for hint in hint_order:                                       # the plan's fallback hint order
        for o in options:
            if o.operation == "retry_current" and o.cue == hint:
                return o
    nexts = [o for o in options if o.operation == "next_exercise"]
    return min(nexts, key=lambda o: abs(o.tier - ctx.current.tier))   # same level when offered


def decide(context: DecisionContext, options: list[PracticeAction], patient_id: str,
           hint_order: list[str] = ("semantic", "first_sound", "sentence_frame", "first_letter")) -> dict:
    version = ACTIVE.get(patient_id, PCFG.default_model)          # "ditto:sft200-g2-s0" or "ditto:<patient run>"
    t0 = time.perf_counter()
    try:
        policy = REGISTRY.policy(version)                         # loads the adapter folder for this version
        text, _ = policy.generate(context, options)               # option scoring with the timeout from the config
        keys = [format_action(o) for o in options]
        if text not in keys:
            raise ValueError("invalid output")
        return {"action": json.loads(text), "source": version, "fallback_reason": None, "raw_output": text,
                "latency_ms": round((time.perf_counter() - t0) * 1000)}
    except concurrent.futures.TimeoutError:
        reason = "timeout"
    except Exception as e:                                        # model unavailable, invalid output, ...
        reason = f"{type(e).__name__}: {str(e)[:200]}"
    a = fallback_action(context, options, list(hint_order))
    return {"action": json.loads(format_action(a)), "source": "fallback", "fallback_reason": reason,
            "raw_output": None, "latency_ms": round((time.perf_counter() - t0) * 1000)}
```

## 6. Golden test

`docs/ditto_golden_test.json` holds 5 situations (DecisionContext plus the options the server offers) and
the exact action the current code returns for each, with the per-option log-probabilities. Check a port:

```bash
uv run --extra llm python -m scripts.ditto_golden_check
# g1 ... OK  ...  golden test passed: all 5 decisions identical
```

| case | situation | options offered | reference action | demonstrator prefers |
|---|---|---|---|---|
| g1 | wrong word on the first try of "banana", tries left | the 4 hints (decision 1d) | `{"op":"retry","hint":"semantic"}` | same |
| g2 | 2 failed tries on "glasses", meaning hint already shown | first sound, first letter, sentence | `{"op":"retry","hint":"first_sound"}` | same |
| g3 | "cat" correct without a hint in 2.1 s, tier 1 | next tier 1, next tier 2 | `{"op":"next","tier":2}` | same |
| g4 | third try on "kangaroo" (tier 3) failed, no tries left | next tier 2, next tier 3 | `{"op":"next","tier":3}` | `{"op":"next","tier":2}` (rule F) |
| g5 | sound error on "chair" after 24 minutes | 4 hints and next tiers 1-3 | `{"op":"retry","hint":"first_sound"}` | `{"op":"next","tier":1}` (rule G) |

g4 and g5 are deliberate: they show the two weak rules (section 7). A faithful port must reproduce the
reference action, not the demonstrator's. Case g1 in full:

```json
{"id": "g1-meaning-error-first-try",
 "context": {"current": {"exercise_id": "ex-banana", "word": "banana", "tier": 2, "hints_shown": [], "attempts": 1},
             "history": [{"exercise_id": "ex-cat", "word": "cat", "tier": 1, "hint": "none", "correct": true, "error": null, "response_time_s": 2.4},
                         {"exercise_id": "ex-banana", "word": "banana", "tier": 2, "hint": "none", "correct": false, "error": "meaning", "response_time_s": 4.2}],
             "session_minutes": 6},
 "options": ["{\"op\":\"retry\",\"hint\":\"first_sound\"}", "{\"op\":\"retry\",\"hint\":\"first_letter\"}",
             "{\"op\":\"retry\",\"hint\":\"sentence_frame\"}", "{\"op\":\"retry\",\"hint\":\"semantic\"}"],
 "expected": {"action": "{\"op\":\"retry\",\"hint\":\"semantic\"}"}}
```

Log-probabilities may differ slightly between devices (MPS, CUDA); the chosen action must not.

## 7. Known limitations

- **Rule G (error at 20+ minutes -> easier picture) and rule F (still wrong after two hints -> easier)** are
  weak: 27% and 33% on test (golden cases g5 and g4). Few-shot prompting does better on both.
- **Decision 1d (hints only while tries are left)** removes "next" options after a wrong answer with tries
  left. The model was trained with those options present and still ranks the remaining hints correctly
  (g1, g2), but rule G can then only act after the last try or a correct answer.
- **DITTO vs SFT is a tie** (75% vs 77%, difference -1.8 points, CI -7.3 to +4.2). DITTO was chosen on dev
  (29/33 vs 25/33), but that lead did not carry over to test. Both are far above the untrained model (+46
  points) and prompting (+37).
- **The demonstrator is fictional**: the demonstrations were generated from demonstrator A's written
  strategy and are pending human review. They are not clinical knowledge, and the results say nothing about
  patients.
- **Per-patient adapters are in progress.** The trainer can start from the shared adapter
  (`init_adapter`), but trained on a few clinician examples alone it over-generalised
  (`results/patient_steering/README_score.md`). The fix with anchors (the shared adapter's own choices
  elsewhere, as in `service/live_ditto.py`) is not finished, and the section 8 preview numbers are not
  implemented yet.
- **The shared adapter already chooses the first-sound hint after a failed meaning hint** (g2), so a demo of
  steering needs a preference it does not already follow, for example the first-letter hint there.
