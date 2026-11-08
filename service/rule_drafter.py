"""Turns a clinician's note into a rule (docs/API_CONTRACT.md §7, decision 1b) with the Claude API.

    drafter.draft("When meaning hints have repeatedly failed, try a first-sound hint ...") -> (Rule, None)
    drafter.draft("She seems tired on Mondays.")                                           -> (None, "why not")

The key is read from ANTHROPIC_API_KEY (the gitignored .env is loaded first) on the server only; it is never
sent to the browser. Model: claude-opus-5-5, or $CUEAGAIN_RULE_MODEL. Claude answers in a flat structured form
(`RuleDraft`), which `to_rule` converts to the contract's `Rule` and validates. Anything that goes wrong (no
key, a refusal, an API error, output that doesn't validate, a note outside what a rule can express) gives
(None, problem), and the clinician fills in the rule by hand. A note is never turned into a rule silently:
the result is a draft until the clinician confirms it.
"""
from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel, ValidationError

from core.config import load_dotenv
from core.contract import Hint, Range, RetryThen, NextThen, Rule, RuleWhen

DEFAULT_MODEL = "claude-opus-5-5"
NOT_CONFIGURED = "Rule drafting is not configured (no ANTHROPIC_API_KEY); fill in the rule by hand."

SYSTEM_PROMPT = """\
You turn a speech-language pathologist's note about one patient into a structured rule for a picture-naming \
practice app. The app shows a picture; the patient names it. After every counted answer the app's decision model \
chooses the next step, and a rule tells it what the clinician wants in a given situation.

What a rule can say. Every "when" field is optional; all the fields you give must hold at the same time.
- last_error: list of values for the patient's latest counted answer on this picture. Values: "correct", \
"sound" (a sound error: a word that sounds like the target), "meaning" (a wrong word), "no_response" (said \
"I don't know" or pressed "I'm not sure"), "unclassified".
- min_failed_attempts_on_picture / max_failed_attempts_on_picture: how many incorrect counted answers the \
patient has given on this picture so far (including the latest one).
- hints_tried_include: hints that were already shown on this picture. hints_tried_exclude: hints not shown yet.
- min_missed_pictures_in_a_row: how many pictures in a row just before this one ended without a correct answer.
- level: list of difficulty levels (1 easiest, 2, 3 hardest) of the current picture.
The four hints, and what they mean:
- "first_sound": the first sound of the word is spoken ("It starts with the sound k...").
- "first_letter": the first letter is shown.
- "sentence_frame": a sentence with the word missing is spoken ("A pet that says meow is a ___").
- "semantic": the meaning of the word is described, a "meaning hint" ("A small furry pet that purrs").

What a rule can ask for (then_op):
- "retry": show the same picture again with one hint (then_hint = one of the four hints). The app never repeats a \
hint on the same picture, and a patient has at most 3 tries per picture.
- "next": move on to another picture, easier / same / harder than the current one (then_level).

Notes can steer hints and difficulty only. Anything else cannot be a rule: scheduling, session length, the \
patient's mood, specific pictures or words, how the app looks, praise or wording, medical advice, or a note that \
is only an observation. For those, return understood=false and put a short, plain explanation in "problem" \
(for example: "Notes can steer hints and difficulty only"). Also return understood=false if the note is too vague \
to turn into conditions and an action, or if it contradicts itself. Do not guess: a wrong rule is worse than none.

When understood=true:
- Set only the "when" fields the note actually implies, and leave the others null.
- "repeatedly failed" or "keeps getting it wrong" means at least 2 failed attempts on the picture, unless the \
note gives a number.
- "meaning hints have failed" means hints_tried_include contains "semantic" and the patient is still failing.
- If the note says what to try "before moving to another picture", the action is a retry with that hint.
- "summary" is one plain sentence a clinician can check, such as "After 2 or more failed tries on a picture \
where a meaning hint was already given → give the first-sound hint."
- problem must be null.
When understood=false: leave every other field null except "summary", which can be an empty string."""


class RuleDraft(BaseModel):
    """Claude's answer: a flat form (no unions) that `to_rule` turns into a contract `Rule`."""
    understood: bool
    problem: str | None = None
    last_error: list[Literal["correct", "sound", "meaning", "no_response", "unclassified"]] | None = None
    min_failed_attempts_on_picture: int | None = None
    max_failed_attempts_on_picture: int | None = None
    hints_tried_include: list[Hint] | None = None
    hints_tried_exclude: list[Hint] | None = None
    min_missed_pictures_in_a_row: int | None = None
    level: list[int] | None = None
    then_op: Literal["retry", "next"] | None = None
    then_hint: Hint | None = None
    then_level: Literal["easier", "same", "harder"] | None = None
    summary: str = ""


def _range(lo: int | None, hi: int | None) -> Range | None:
    return None if lo is None and hi is None else Range(min=lo, max=hi)


def to_rule(d: RuleDraft) -> tuple[Rule | None, str | None]:
    """Convert and validate. Returns (rule, None), or (None, why not)."""
    if not d.understood:
        return None, (d.problem or "This note can't be turned into a rule about hints or difficulty.").strip()
    try:
        when = RuleWhen(
            last_error=d.last_error or None,
            failed_attempts_on_picture=_range(d.min_failed_attempts_on_picture, d.max_failed_attempts_on_picture),
            hints_tried_include=d.hints_tried_include or None, hints_tried_exclude=d.hints_tried_exclude or None,
            missed_pictures_in_a_row=_range(d.min_missed_pictures_in_a_row, None),
            level=d.level or None)
        if d.then_op == "retry" and d.then_hint:
            then = RetryThen(op="retry", hint=d.then_hint)
        elif d.then_op == "next" and d.then_level:
            then = NextThen(op="next", level=d.then_level)
        else:
            return None, "The drafted rule didn't say what to do (a hint, or an easier/same/harder picture)."
        rule = Rule(when=when, then=then, summary=d.summary.strip())
    except (ValidationError, ValueError) as e:
        return None, f"The drafted rule was not valid ({str(e).splitlines()[0][:150]})."
    if not d.summary.strip():
        return None, "The drafted rule had no summary."
    if when == RuleWhen():
        return None, "The note doesn't say when the rule applies; fill in the conditions by hand."
    from service.steering import check_rule
    from service.errors import Invalid
    try:
        check_rule(rule)
    except Invalid as e:
        return None, f"The drafted rule was not usable ({e})."
    return rule, None


class ClaudeRuleDrafter:
    def __init__(self, model: str | None = None, client=None) -> None:
        self.model, self.client = model, client

    def draft(self, note_text: str) -> tuple[Rule | None, str | None]:
        load_dotenv()
        key = os.environ.get("ANTHROPIC_API_KEY")
        client = self.client
        if client is None:
            if not key:
                return None, NOT_CONFIGURED
            import anthropic
            client = anthropic.Anthropic(api_key=key)
        model = self.model or os.environ.get("CUEAGAIN_RULE_MODEL") or DEFAULT_MODEL
        try:
            response = client.messages.parse(model=model, max_tokens=4000, system=SYSTEM_PROMPT,
                                             messages=[{"role": "user", "content": note_text}],
                                             output_format=RuleDraft)
            if response.stop_reason == "refusal":
                return None, "Claude declined to draft a rule for this note; fill in the rule by hand."
            parsed = response.parsed_output
            if parsed is None:
                return None, "Claude's answer could not be read; fill in the rule by hand."
            return to_rule(parsed)
        except Exception as e:                      # any API or parsing problem: no rule, never a crash
            return None, f"Rule drafting failed ({type(e).__name__}: {str(e)[:120]}); fill in the rule by hand."
