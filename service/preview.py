"""The preview of a finished training run (docs/API_CONTRACT.md §8): the new adapter against the patient's
current model, as counts with totals. Nothing here is scripted: every number is a count of real model choices.

  follows_your_examples              the run's own examples: how often each model picks the clinician's action.
                                     They were trained on, so this shows the update took, not that it generalises.
  follows_your_rules_new_situations  for each confirmed rule of this patient, up to 10 situations built by code
                                     from the rule's `when` fields (other pictures, varied history); how often each
                                     model picks the rule's action. The situations are generated, not real sessions.
  unchanged_elsewhere                the 10 held-out test situations of data/demonstrations/demonstrator_a_v1.jsonl,
                                     with the live app's option rule (hints only while tries are left): how often
                                     the new adapter chooses what the current model chooses.

The models are asked through the same decider as the live app (agents.registry, with the plan's hint order as the
fallback when a model's output is invalid), so a fallback counts as the choice the app would have made.
"""
from __future__ import annotations

import json
import random
import zlib
from collections.abc import Callable
from types import SimpleNamespace
from typing import TYPE_CHECKING

from core.config import resolve
from core.contract import (
    HINTS, Action, Agreement, ContractContext, ModelDecision, NextAction, Preview, RetryAction, Rule, Unchanged,
    matches_rule, rule_action,
)
from core.practice import AttemptSummary, ItemBank, parse_action
from core.schema import CurrentExercise
from service.engine import decision_options, plan_order_action

if TYPE_CHECKING:
    from service.engine import Engine
    from service.training import TrainingRec

TEST_ROWS = "data/demonstrations/demonstrator_a_v1.jsonl"
PER_RULE = 10
EXAMPLES_LABEL = "trained on these: shows the update took, not evidence"
RULES_LABEL = "situations generated from your rules, on other pictures; not real sessions"


def _to_contract(a) -> Action:
    return RetryAction(op="retry", hint=a.cue) if a.operation == "retry_current" else NextAction(op="next", tier=a.tier)


def hint_only(options: list[Action], hints_shown: list[str]) -> list[Action]:
    """The live app's option rule for stored test situations: no hint twice on a picture, and hints only while any are left."""
    opts = [o for o in options if not (isinstance(o, RetryAction) and o.hint in hints_shown)]
    retries = [o for o in opts if isinstance(o, RetryAction)]
    return retries or opts


def held_out_situations(patient_id: str, path: str = TEST_ROWS) -> list[tuple[ContractContext, list[Action]]]:
    rows = [json.loads(line) for line in resolve(path).read_text().splitlines() if line.strip()]
    out = []
    for r in rows:
        if r["split"] != "test":
            continue
        ctx = ContractContext(patient_id=patient_id, **r["context"])
        offered = [parse_action(o) for o in r["options"]]
        offered = [o for o in offered if not (o.operation == "retry_current" and o.cue == "none")]   # the app never offers a plain retry
        opts = hint_only([_to_contract(o) for o in offered], ctx.current.hints_shown)
        out.append((ctx, opts))
    return out


_ERRORS = ["sound", "meaning", "no_response", "unclassified"]


def generate_situations(rule: Rule, bank: ItemBank, max_attempts: int, avoid: set[str], n: int, rng: random.Random,
                        patient_id: str = "preview", tries: int = 5000) -> list[tuple[ContractContext, list[Action]]]:
    """Up to `n` distinct, code-generated situations that satisfy `rule.when` and where the rule's action is one of
    the options the live app would offer. Pictures in `avoid` (those the rule was made from) are used last."""
    w = rule.when
    items = [i for i in bank.items if w.level is None or i.tier in w.level]
    fresh = [i for i in items if i.exercise_id not in avoid] or items
    out, seen = [], set()
    for _ in range(tries):
        if len(out) >= n or not items:
            break
        item = rng.choice(fresh)
        last = rng.choice(w.last_error) if w.last_error else rng.choice(_ERRORS)
        fr = w.failed_attempts_on_picture
        lo = max(0 if last == "correct" else 1, (fr.min or 0) if fr else 0)
        hi = min(max_attempts, fr.max if fr and fr.max is not None else max_attempts)
        if lo > hi:
            continue
        failed = rng.randint(lo, hi)
        attempts = failed + (1 if last == "correct" else 0)
        h = attempts - 1                                         # every attempt after the first had a hint
        inc, exc = list(w.hints_tried_include or []), set(w.hints_tried_exclude or [])
        rest = [x for x in HINTS if x not in inc and x not in exc]
        if len(inc) > h or h - len(inc) > len(rest):
            continue
        seq = inc + rng.sample(rest, h - len(inc))
        rng.shuffle(seq)
        picture = []
        for k in range(attempts):
            is_last = k == attempts - 1
            ok = is_last and last == "correct"
            err = None if ok else (last if is_last else rng.choice(_ERRORS))
            picture.append(AttemptSummary(exercise_id=item.exercise_id, word=item.preferred_name, tier=item.tier,
                                          hint="none" if k == 0 else seq[k - 1], correct=ok, error=err,
                                          response_time_s=round(rng.uniform(2, 7), 1)))
        mr = w.missed_pictures_in_a_row
        m_lo = (mr.min or 0) if mr else 0
        m = rng.randint(m_lo, mr.max if mr and mr.max is not None else m_lo + 2)
        others = [i for i in bank.items if i.exercise_id != item.exercise_id]
        before: list[AttemptSummary] = []
        prior = rng.sample(others, min(len(others), m + (rng.randint(0, 2) if m else rng.randint(1, 2))))
        for j, o in enumerate(prior):
            missed = j >= len(prior) - m                          # the last m prior pictures were missed
            for k in range(max_attempts if missed else 1):
                right = not missed
                before.append(AttemptSummary(exercise_id=o.exercise_id, word=o.preferred_name, tier=o.tier, hint="none",
                                             correct=right, error=None if right else rng.choice(_ERRORS),
                                             response_time_s=round(rng.uniform(2, 7), 1)))
        ctx = ContractContext(
            patient_id=patient_id, history=before + picture, session_minutes=round(rng.uniform(1, 9), 1),
            current=CurrentExercise(exercise_id=item.exercise_id, word=item.preferred_name, tier=item.tier,
                                    hints_shown=seq[:h], attempts=attempts))
        opts = decision_options(ctx, ctx.current.hints_shown, max_attempts)
        key = (item.exercise_id, tuple(seq), last, failed, m)
        if key in seen or not matches_rule(rule, ctx) or rule_action(rule, ctx) not in opts:
            continue
        seen.add(key)
        out.append((ctx, opts))
    return out


class ModelPreviewer:
    """`decide` is the live app's decider (service.patient_api.model_decider): decide(ctx, options, session)."""

    def __init__(self, decide: Callable, per_rule: int = PER_RULE, test_path: str = TEST_ROWS) -> None:
        self.decide, self.per_rule, self.test_path = decide, per_rule, test_path

    def _choose(self, version: str, plan, ctx: ContractContext, options: list[Action]) -> Action:
        s = SimpleNamespace(plan=plan, session_id="preview", history=ctx.history, model_version=version)
        try:
            md: ModelDecision = self.decide(ctx, options, s)
            if md.action in options:
                return md.action
        except Exception:
            pass
        return plan_order_action(plan, ctx, options)             # what the app does when a model fails

    def __call__(self, engine: "Engine", rec: "TrainingRec") -> Preview:
        pid = rec.patient_id
        patient = engine.patient(pid)
        new_v = f"ditto:{rec.run_id}"
        cur_v = engine.training.adapter_version(patient.active_model.version) if engine.training else patient.active_model.version
        pick = lambda v, ctx, opts: self._choose(v, patient.plan, ctx, opts)    # noqa: E731

        ex_new = ex_cur = 0
        examples = [e for e in engine.steering.examples(pid) if e.example_id in rec.example_ids]
        for e in examples:
            _, _, d = engine.steering.find_decision(e.decision_id)
            ex_new += pick(new_v, d.context, d.options) == e.preferred
            ex_cur += pick(cur_v, d.context, d.options) == e.preferred

        r_new = r_cur = r_total = 0
        for note in engine.steering.notes(pid):
            if note.rule_status != "confirmed" or note.rule is None:
                continue
            avoid = {d.context.current.exercise_id for _, _, d in engine.steering.patient_decisions(pid)
                     if matches_rule(note.rule, d.context)}
            rng = random.Random(zlib.crc32(f"{rec.run_id}:{note.note_id}".encode()))
            for ctx, opts in generate_situations(note.rule, engine.bank, patient.plan.max_attempts, avoid, self.per_rule,
                                                 rng, pid):
                want = rule_action(note.rule, ctx)
                r_total += 1
                r_new += pick(new_v, ctx, opts) == want
                r_cur += pick(cur_v, ctx, opts) == want

        same = total = 0
        for ctx, opts in held_out_situations(pid, self.test_path):
            total += 1
            same += pick(new_v, ctx, opts) == pick(cur_v, ctx, opts)

        return Preview(
            follows_your_examples=Agreement(new=ex_new, current=ex_cur, total=len(examples), label=EXAMPLES_LABEL),
            follows_your_rules_new_situations=Agreement(
                new=r_new, current=r_cur, total=r_total,
                label=RULES_LABEL if r_total else "no confirmed rule produced new situations to test"),
            unchanged_elsewhere=Unchanged(same=same, total=total))
