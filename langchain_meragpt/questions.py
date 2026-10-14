"""Builders for Decider questions, and helpers for reading answers.

Three question types:

- `yes_no` (API type `noul`): the answer is the probability of yes.
- `choice`: pick one of up to 10 named options; the answer is the chosen label
  and a probability for every label.
- `score`: place the state on an ordered scale (lowest level first); the answer
  is the expected level and a probability for every level.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

MAX_OPTIONS = 10
MAX_QUESTIONS = 64


def yes_no(instructions: str) -> dict[str, Any]:
    """A yes/no question. Its answer is `{"type": "noul", "noul": p_yes}`."""
    return {"type": "noul", "instructions": instructions}


def choice(instructions: str, options: Mapping[str, str | None] | Sequence[str]) -> dict[str, Any]:
    """Pick one option.

    Args:
        instructions: The question.
        options: Labels, or a mapping of label to a one-line description of
            when it applies. Descriptions noticeably improve accuracy.
    """
    criteria = dict(options) if isinstance(options, Mapping) else {str(o): None for o in options}
    if not 1 <= len(criteria) <= MAX_OPTIONS:
        raise ValueError(
            f"A choice takes 1 to {MAX_OPTIONS} options; got {len(criteria)}. "
            'Split a larger set into groups with a "none of these" option.'
        )
    return {"type": "choice", "instructions": instructions, "criteria": criteria}


def score(instructions: str, levels: Sequence[str]) -> dict[str, Any]:
    """Place the state on an ordered scale.

    Args:
        instructions: The question.
        levels: Level descriptions, lowest first.
    """
    levels = [str(x) for x in levels]
    if not 1 <= len(levels) <= MAX_OPTIONS:
        raise ValueError(f"A score takes 1 to {MAX_OPTIONS} levels; got {len(levels)}.")
    return {"type": "score", "instructions": instructions, "criteria": levels}


def probability(answer: Mapping[str, Any], label: str | None = None) -> float:
    """The probability an answer gives to `label`.

    For a yes/no answer, the probability of yes (`label` is ignored). For a
    choice, the probability of `label`, defaulting to the chosen one. For a
    score, the probability of level `label` (`"0"`, `"1"`, ...), defaulting to
    the most likely one.
    """
    if answer.get("type") == "noul":
        return float(answer["noul"])
    probs = answer.get("probabilities") or {}
    if label is None:
        return float(answer.get("confidence", max(probs.values(), default=0.0)))
    return float(probs.get(label, 0.0))
