"""Decision policies for the practice flow, behind one `decide(ctx, options) -> Decision` call, plus
the original `Policy` interface (`act` / `update`) for compatibility with /act.

  LLMDecisionPolicy     a language model (unadapted base, few-shot prompted base, or base + a trained
                        LoRA adapter) chooses one of the valid options.
  BanditDecisionPolicy  the existing Beta-Bernoulli Thompson bandit, mapped onto practice options.
                        A baseline: it learns tier/hint success rates and ignores word identity,
                        error category and response time.

Every model output is validated against the valid options at this boundary. An unparseable,
invalid or timed-out answer triggers an explicit fallback (the bandit baseline), and the returned
Decision then reports `source` = the fallback and `fallback_reason`, never the language model.
"""
from __future__ import annotations

import concurrent.futures
import time

from pydantic import BaseModel

from core.config import BetaBanditConfig, ClinicalConfig
from core.masks import action_mask
from core.practice import (DecisionContext, PracticeAction, format_action, from_tier_cue, messages,
                           parse_action, to_state)
from core.schema import Action, Outcome, State


class Decision(BaseModel):
    action: PracticeAction
    source: str                       # model version that actually produced `action`
    requested_model: str              # model version that was asked
    model_detail: dict                # base model, revision, adapter path, ...
    fallback_reason: str | None = None
    raw_output: str | None = None
    parsed: bool | None = None        # language models only: the output parsed as an action
    valid: bool | None = None         # ... and was one of the valid options
    latency_ms: float
    options: list[str]


class BanditDecisionPolicy:
    """Beta-Bernoulli Thompson bandit (agents/beta_bandit.py) over tier x hint, mapped to options."""

    def __init__(self, bcfg: BetaBanditConfig, clinical: ClinicalConfig, seed: int, version: str = "bandit") -> None:
        from agents.beta_bandit import BetaThompsonBandit
        self.bandit = BetaThompsonBandit(bcfg, clinical, seed=seed)
        self.clinical, self.version = clinical, version
        self.name = version

    def observe(self, state: State, shown_tier: int, shown_cue: str, outcome: Outcome) -> None:
        self.bandit.update(state, Action(tier=shown_tier, cue=shown_cue), outcome)

    def decide(self, ctx: DecisionContext, options: list[PracticeAction], session_id: str | None = None,
               trial_count: int = 0, requested: str | None = None, reason: str | None = None,
               extra: dict | None = None) -> Decision:
        t0 = time.perf_counter()
        state = to_state(ctx, session_id, trial_count)
        mask = action_mask(state.current_tier, self.clinical.approved_cues, self.clinical.max_tier_step)
        a = self.bandit.act(state, mask)
        action = from_tier_cue(a.tier, a.cue, options)
        return Decision(action=action, source=self.version, requested_model=requested or self.version,
                        model_detail={"type": "beta_bandit", "mapped_from": {"tier": a.tier, "cue": a.cue},
                                      **(extra or {})},
                        fallback_reason=reason, latency_ms=(time.perf_counter() - t0) * 1000,
                        options=[format_action(o) for o in options])


class LLMDecisionPolicy:
    def __init__(self, runtime, version: str, adapter: str | None, detail: dict, history_in_prompt: int,
                 timeout_s: float, few_shot: list[tuple[str, str]] | None = None,
                 executor: concurrent.futures.Executor | None = None, prompt_format: str = "v1") -> None:
        self.runtime, self.version, self.adapter, self.detail = runtime, version, adapter, detail
        self.prompt_format = prompt_format
        self.history_in_prompt, self.timeout_s, self.few_shot = history_in_prompt, timeout_s, few_shot
        self.executor = executor or concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.name = version

    def generate(self, ctx: DecisionContext, options: list[PracticeAction]) -> tuple[str, float]:
        msgs = messages(ctx, options, self.history_in_prompt, self.few_shot, self.prompt_format)
        if getattr(getattr(self.runtime, "cfg", None), "decoding", "generate") == "score":
            keys = [format_action(o) for o in options]
            fut = self.executor.submit(self.runtime.score, msgs, keys, self.adapter)
            lps, dt = fut.result(timeout=self.timeout_s)
            return keys[max(range(len(keys)), key=lps.__getitem__)], dt
        fut = self.executor.submit(self.runtime.generate, msgs, self.adapter)
        return fut.result(timeout=self.timeout_s)

    def decide(self, ctx: DecisionContext, options: list[PracticeAction], fallback: BanditDecisionPolicy,
               session_id: str | None = None, trial_count: int = 0) -> Decision:
        keys = [format_action(o) for o in options]
        t0 = time.perf_counter()
        try:
            text, _ = self.generate(ctx, options)
        except concurrent.futures.TimeoutError:
            return self._fallback(ctx, options, fallback, session_id, trial_count, None, None, None,
                                  f"timeout after {self.timeout_s:.0f} s", t0)
        except Exception as e:      # model failed to load or run: fall back, and say why
            return self._fallback(ctx, options, fallback, session_id, trial_count, None, None, None,
                                  f"model error: {type(e).__name__}: {str(e)[:200]}", t0)
        parsed = parse_action(text)
        if parsed is None:
            return self._fallback(ctx, options, fallback, session_id, trial_count, text, False, False,
                                  "output could not be parsed as an action", t0)
        if parsed.key() not in keys:
            return self._fallback(ctx, options, fallback, session_id, trial_count, text, True, False,
                                  f"output {parsed.key()} is not one of the valid options", t0)
        return Decision(action=parsed, source=self.version, requested_model=self.version, model_detail=self.detail,
                        raw_output=text, parsed=True, valid=True,
                        latency_ms=(time.perf_counter() - t0) * 1000, options=keys)

    def _fallback(self, ctx, options, fallback, session_id, trial_count, text, parsed, valid, reason, t0) -> Decision:
        d = fallback.decide(ctx, options, session_id, trial_count, requested=self.version, reason=reason,
                            extra={"fallback_for": self.version})
        return d.model_copy(update={"raw_output": text, "parsed": parsed, "valid": valid,
                                    "latency_ms": (time.perf_counter() - t0) * 1000})


def context_from_state(state: State) -> DecisionContext:
    """Rebuild a DecisionContext from an original-contract State that carries `current_exercise`."""
    from core.practice import AttemptSummary
    if state.current_exercise is None or not state.last_trials:
        raise ValueError("a language-model decision needs current_exercise and at least one trial")
    hist = [AttemptSummary(exercise_id=t.exercise_id or f"ex-{t.word}", word=t.word, tier=t.tier, hint=t.cue,
                           correct=t.outcome, error=None if t.outcome else t.error_type,
                           response_time_s=t.response_time) for t in state.last_trials]
    return DecisionContext(current=state.current_exercise, history=hist, session_minutes=state.session_minutes)


def to_tier_cue(action: PracticeAction, current_tier: int) -> tuple[int, str]:
    if action.operation == "retry_current":
        return current_tier, action.cue
    return int(action.tier), "none"



class LLMActPolicy:
    """The original Policy interface (act / update) around a language-model decision policy, for
    POST /act with `agent: llm`. The State must carry `current_exercise`; otherwise, and on any
    invalid model output, the session's bandit answers and `last_decision` says so. `update` only
    feeds the bandit fallback: the language model is not fine-tuned on patient answers."""

    def __init__(self, registry, version: str, pcfg, clinical: ClinicalConfig, bandit, bcfg: BetaBanditConfig) -> None:
        self.registry, self.version, self.pcfg, self.clinical = registry, version, pcfg, clinical
        self.bandit = bandit
        self.name = f"llm:{version}"
        self.last_decision: Decision | None = None

    def act(self, state: State, mask) -> Action:
        from core.practice import valid_options
        try:
            ctx = context_from_state(state)
        except ValueError as e:
            a = self.bandit.act(state, mask)
            self.last_decision = Decision(action=from_tier_cue(a.tier, a.cue, [
                PracticeAction(operation="next_exercise", tier=a.tier)]), source="bandit", requested_model=self.version,
                model_detail={}, fallback_reason=str(e), latency_ms=0.0, options=[])
            return a.model_copy(update={"rationale": f"bandit fallback: {e}"[:500]})
        opts = valid_options(ctx, self.clinical.approved_cues, self.clinical.max_tier_step,
                             self.pcfg.max_attempts_per_exercise)
        fb = _WrappedBandit(self.bandit, self.clinical)
        try:
            pol = self.registry.policy(self.version)
            d = pol.decide(ctx, opts, fb, state.session_id, state.trial_count)
        except Exception as e:
            d = fb.decide(ctx, opts, state.session_id, state.trial_count, requested=self.version,
                          reason=f"model unavailable: {type(e).__name__}: {str(e)[:200]}")
        self.last_decision = d
        tier, cue = to_tier_cue(d.action, ctx.current.tier)
        why = f"{d.source}: {d.action.operation}" + (f" (fallback: {d.fallback_reason})" if d.fallback_reason else "")
        return Action(tier=tier, cue=cue, rationale=why[:500])

    def update(self, state: State, action: Action, outcome: Outcome) -> None:
        self.bandit.update(state, action, outcome)


class _WrappedBandit(BanditDecisionPolicy):
    """BanditDecisionPolicy around an existing bandit instance (the session's)."""

    def __init__(self, bandit, clinical: ClinicalConfig) -> None:
        self.bandit, self.clinical, self.version, self.name = bandit, clinical, "bandit", "bandit"
