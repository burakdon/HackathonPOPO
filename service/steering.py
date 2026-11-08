"""Clinician steering (docs/API_CONTRACT.md §7): corrections, notes, rules, their matches and the training
examples they produce. Part of the practice engine (service/engine.py), stored as JSON under the same root:

  examples/<patient_id>.json   {counter, examples}      example IDs x-0001, ... per patient
  notes/<patient_id>.json      {notes}                  note IDs n-0001, ... across all patients

An example is a decision's stored context plus the action the clinician prefers. It stays "pending" until a
training run that includes it completes, then reads "used in <run_id>". A decision has at most one example:
choosing again while it is pending replaces it; once it was used in training the answer is 409.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, Field

from core.contract import (
    Action, CorrectionIn, Example, Note, Range, Rule, RuleMatch, StoredDecision, matches_rule, rule_action,
    situation,
)
from service.errors import Conflict, Invalid, NotFound

if TYPE_CHECKING:
    from service.engine import Engine, PresRec, SessionRec


class RuleDrafter(Protocol):
    """Turns a clinician's note into a rule. Returns (rule, None) or (None, why not). Never raises."""

    def draft(self, note_text: str) -> tuple[Rule | None, str | None]: ...


class ExamplesRec(BaseModel):
    patient_id: str
    counter: int = 0
    examples: list[Example] = Field(default_factory=list)


class NoteRec(BaseModel):
    note: Note
    drafted: Rule | None = None            # what Claude proposed, to tell "claude" from "clinician" on confirmation


class NotesRec(BaseModel):
    patient_id: str
    notes: list[NoteRec] = Field(default_factory=list)


def action_json(a: Action) -> str:
    return json.dumps(a.model_dump(), separators=(",", ":"))


def check_rule(rule: Rule) -> None:
    """Checks beyond the schema: ranges that can never hold, and a rule that says nothing."""
    for name in ("failed_attempts_on_picture", "missed_pictures_in_a_row"):
        r: Range | None = getattr(rule.when, name)
        if r is not None and r.min is not None and r.max is not None and r.min > r.max:
            raise Invalid(f"{name}: min {r.min} is above max {r.max}")
        if r is not None and r.min is None and r.max is None:
            raise Invalid(f"{name}: give a min and/or a max")
    for name in ("last_error", "hints_tried_include", "hints_tried_exclude", "level"):
        v = getattr(rule.when, name)
        if v is not None and not v:
            raise Invalid(f"{name}: give at least one value, or leave it out")
    both = set(rule.when.hints_tried_include or []) & set(rule.when.hints_tried_exclude or [])
    if both:
        raise Invalid(f"hints_tried_include and hints_tried_exclude both name {sorted(both)}")
    if not rule.summary.strip():
        raise Invalid("summary: describe the rule in a sentence")


class Steering:
    def __init__(self, engine: "Engine") -> None:
        self.engine = engine
        self.examples_by_patient: dict[str, ExamplesRec] = {}
        self.notes_by_patient: dict[str, NotesRec] = {}
        for kind, cls, store in (("examples", ExamplesRec, self.examples_by_patient),
                                 ("notes", NotesRec, self.notes_by_patient)):
            d = engine.root / kind
            for p in sorted(d.glob("*.json")) if d.exists() else []:
                try:
                    rec = cls.model_validate_json(p.read_text())
                    store[rec.patient_id] = rec
                except Exception as e:
                    print(f"engine: could not load {p}: {e}")

    # ---- decisions
    def find_decision(self, decision_id: str) -> tuple["SessionRec", "PresRec", StoredDecision]:
        for s in self.engine.sessions.values():
            for pr in s.pictures:
                for d in pr.decisions:
                    if d.decision_id == decision_id:
                        return s, pr, d
        raise NotFound(f"no decision {decision_id!r}")

    def patient_decisions(self, pid: str) -> list[tuple["SessionRec", "PresRec", StoredDecision]]:
        out = []
        for s in sorted((s for s in self.engine.sessions.values() if s.patient_id == pid), key=lambda s: s.started):
            for pr in s.pictures:
                out += [(s, pr, d) for d in pr.decisions]
        return out

    # ---- examples
    def _ex(self, pid: str) -> ExamplesRec:
        return self.examples_by_patient.setdefault(pid, ExamplesRec(patient_id=pid))

    def examples(self, pid: str) -> list[Example]:
        self.engine.patient(pid)
        return list(self._ex(pid).examples)

    def pending(self, pid: str) -> int:
        rec = self.examples_by_patient.get(pid)
        return sum(e.status == "pending" for e in rec.examples) if rec else 0

    def correction_for(self, pid: str, decision_id: str) -> Example | None:
        rec = self.examples_by_patient.get(pid)
        return next((e for e in rec.examples if e.decision_id == decision_id and e.source == "correction"), None) if rec else None

    def _save_examples(self, pid: str) -> None:
        self.engine._write("examples", pid, self._ex(pid))

    def _put(self, d: StoredDecision, pid: str, preferred: Action, source: str, note_id: str | None,
             note: str | None) -> Example:
        """Create the decision's example, or replace the pending one. 409 if it was already trained on."""
        rec = self._ex(pid)
        if preferred not in d.options:
            raise Invalid(f"preferred action {action_json(preferred)} is not one of this decision's options")
        old = next((e for e in rec.examples if e.decision_id == d.decision_id), None)
        locked = self.engine.training.in_flight_examples() if self.engine.training else set()
        if old is not None and (old.status != "pending" or (pid, old.example_id) in locked):
            raise Conflict(f"decision {d.decision_id} already has example {old.example_id} that is "
                           f"{old.status if old.status != 'pending' else 'part of a training run in progress'}; "
                           "it can't be changed")
        if old is None:
            rec.counter += 1
            ex = Example(example_id=f"x-{rec.counter:04d}", patient_id=pid, source=source, decision_id=d.decision_id,
                         note_id=note_id, model_chose=d.action, preferred=preferred, note=note, status="pending")
            rec.examples.append(ex)
        else:
            ex = old.model_copy(update={"source": source, "note_id": note_id, "model_chose": d.action,
                                        "preferred": preferred, "note": note})
            rec.examples[rec.examples.index(old)] = ex
        return ex

    def correct(self, decision_id: str, body: CorrectionIn, who: str = "clinician") -> tuple[Example, bool]:
        """Returns the example and whether it is new (201) or replaced one."""
        with self.engine.lock:
            s, _, d = self.find_decision(decision_id)
            existed = any(e.decision_id == decision_id for e in self._ex(s.patient_id).examples)
            ex = self._put(d, s.patient_id, body.preferred, "correction", None, body.note)
            self._save_examples(s.patient_id)
            return ex, not existed

    def mark_used(self, pid: str, example_ids: list[str], run_id: str) -> None:
        with self.engine.lock:
            rec = self._ex(pid)
            rec.examples = [e.model_copy(update={"status": f"used in {run_id}"}) if e.example_id in example_ids
                            and e.status == "pending" else e for e in rec.examples]
            self._save_examples(pid)

    # ---- notes
    def _notes(self, pid: str) -> NotesRec:
        return self.notes_by_patient.setdefault(pid, NotesRec(patient_id=pid))

    def _note(self, note_id: str) -> tuple[NotesRec, NoteRec]:
        for rec in self.notes_by_patient.values():
            for n in rec.notes:
                if n.note.note_id == note_id:
                    return rec, n
        raise NotFound(f"no note {note_id!r}")

    def notes(self, pid: str) -> list[Note]:
        self.engine.patient(pid)
        return [n.note for n in self._notes(pid).notes]

    def add_note(self, pid: str, text: str, drafter: RuleDrafter | None) -> Note:
        self.engine.patient(pid)
        rule, problem = (None, "Rule drafting is not configured; fill in the rule by hand.") if drafter is None \
            else drafter.draft(text)                 # slow (network): not under the lock
        if rule is not None:
            try:
                check_rule(rule)
            except Invalid as e:
                rule, problem = None, f"The drafted rule was not usable ({e})."
        with self.engine.lock:
            n = 1 + sum(len(r.notes) for r in self.notes_by_patient.values())
            note = Note(note_id=f"n-{n:04d}", patient_id=pid, text=text, rule=rule,
                        rule_source="claude" if rule else "none", rule_status="draft" if rule else "none",
                        problem=None if rule else (problem or "No rule could be drafted from this note."))
            self._notes(pid).notes.append(NoteRec(note=note, drafted=rule))
            self.engine._write("notes", pid, self._notes(pid))
            return note

    def confirm_rule(self, note_id: str, rule: Rule) -> Note:
        with self.engine.lock:
            rec, nr = self._note(note_id)
            check_rule(rule)
            same = nr.drafted is not None and rule == nr.drafted
            nr.note = nr.note.model_copy(update={"rule": rule, "rule_status": "confirmed", "problem": None,
                                                 "rule_source": "claude" if same else "clinician"})
            self.engine._write("notes", rec.patient_id, rec)
            self.engine._log(rec.patient_id, "clinician", f"Rule confirmed from note: “{nr.note.text}”")
            return nr.note

    def matches(self, note_id: str) -> list[RuleMatch]:
        with self.engine.lock:
            rec, nr = self._note(note_id)
            if nr.note.rule_status != "confirmed" or nr.note.rule is None:
                raise Conflict(f"note {note_id} has no confirmed rule yet")
            return self._matches(rec.patient_id, nr.note.rule)

    def _matches(self, pid: str, rule: Rule) -> list[RuleMatch]:
        out = []
        for _, _, d in self.patient_decisions(pid):
            if not matches_rule(rule, d.context):
                continue
            want = rule_action(rule, d.context)
            ok = want in d.options
            out.append(RuleMatch(decision_id=d.decision_id, date=d.at[:10], word=d.context.current.word,
                                 situation=situation(d.context), model_chose=d.action, rule_says=want if ok else None,
                                 already_agrees=ok and want == d.action, allowed=ok))
        return out

    def examples_from_note(self, note_id: str, decision_ids: list[str]) -> list[Example]:
        with self.engine.lock:
            rec, nr = self._note(note_id)
            if nr.note.rule_status != "confirmed" or nr.note.rule is None:
                raise Conflict(f"note {note_id} has no confirmed rule yet")
            pid = rec.patient_id
            allowed = {m.decision_id: m for m in self._matches(pid, nr.note.rule) if m.allowed}
            bad = [i for i in dict.fromkeys(decision_ids) if i not in allowed]
            if bad:
                raise Invalid(f"not an allowed match of this note: {', '.join(bad)}")
            snapshot = json.dumps(self._ex(pid).model_dump(mode="json"))
            try:
                out = [self._put(self.find_decision(i)[2], pid, allowed[i].rule_says, "note", note_id, nr.note.text)
                       for i in dict.fromkeys(decision_ids)]
            except Exception:                         # all or nothing
                self.examples_by_patient[pid] = ExamplesRec.model_validate_json(snapshot)
                raise
            self._save_examples(pid)
            return out
