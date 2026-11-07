"""Prompt format and action parsing shared by every language-model therapist (GRPO-trained model,
prompted frontier baseline, live LLM agent).

The prompt contains the State JSON and the list of allowed actions (the mask), so the model never
needs to infer the constraints. The answer must be one JSON object {"tier": int, "cue": str}; an
optional short "rationale" is allowed. Anything else counts as a format failure.
"""
from __future__ import annotations

import json
import re

import numpy as np

from core.masks import ACTIONS
from core.schema import State

SYSTEM_PROMPT = (
    "You choose the next picture-naming trial for a person practising word retrieval after a stroke "
    "(aphasia). Pick a difficulty tier (1 easiest to 3 hardest) and a cue (hint) so the person stays "
    "near 85% correct: hard enough to learn, easy enough not to get frustrated. Prefer the hardest tier "
    "and the weakest cue that keep them near that target. Cues from weakest to strongest: none, "
    "first_letter, semantic or sentence_frame, first_sound. You may only choose an allowed action. "
    'Reply with JSON only, for example {"tier": 2, "cue": "none"}.'
)

_JSON = re.compile(r"\{[^{}]*\}")


def user_prompt(state: State, mask: np.ndarray) -> str:
    allowed = [{"tier": t, "cue": c} for (t, c), ok in zip(ACTIONS, mask) if ok]
    s = state.model_dump(exclude={"session_id"})
    return (f"Session state:\n{json.dumps(s, separators=(',', ':'))}\n\n"
            f"Allowed actions:\n{json.dumps(allowed, separators=(',', ':'))}\n\nNext trial (JSON only):")


def messages(state: State, mask: np.ndarray) -> list[dict[str, str]]:
    return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user_prompt(state, mask)}]


def parse_action(text: str) -> tuple[int, str] | None:
    """First JSON object in the text with a valid tier and cue, else None (format failure)."""
    for m in _JSON.finditer(text):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        tier, cue = obj.get("tier"), obj.get("cue")
        if isinstance(tier, int) and (tier, cue) in ACTIONS:
            return tier, cue
    return None
