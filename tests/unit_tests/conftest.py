from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from langchain_meragpt import MeragptAPIError, MeragptClient


def answer_for(q: Mapping[str, Any], p: float) -> dict[str, Any]:
    if q["type"] == "noul":
        return {"type": "noul", "noul": p}
    if q["type"] == "choice":
        labels = list(q["criteria"])
        probs = {
            lab: (p if i == 0 else (1 - p) / max(len(labels) - 1, 1))
            for i, lab in enumerate(labels)
        }
        top = max(probs, key=lambda k: probs[k])
        return {"type": "choice", "choice": top, "confidence": probs[top], "probabilities": probs}
    levels = q["criteria"]
    probs = {str(i): 1 / len(levels) for i in range(len(levels))}
    return {
        "type": "score",
        "score": (len(levels) - 1) / 2,
        "confidence": 1 / len(levels),
        "legend": {str(i): lv for i, lv in enumerate(levels)},
        "probabilities": probs,
    }


class FakeClient(MeragptClient):
    """Records calls; answers every question with probability `p` (or raises)."""

    def __init__(self, p: float = 0.9, fail: bool = False) -> None:
        super().__init__(api_key="mg-live-test")
        self.p = p
        self.fail = fail
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def decide(
        self, state: Any, questions: Mapping[str, Any], *, model: str | None = None
    ) -> dict[str, Any]:
        self.calls.append((state, dict(questions)))
        if self.fail:
            raise MeragptAPIError("boom", status=503)
        return {"answers": {k: answer_for(q, self.p) for k, q in questions.items()}, "usage": {}}

    async def adecide(
        self, state: Any, questions: Mapping[str, Any], *, model: str | None = None
    ) -> dict[str, Any]:
        return self.decide(state, questions, model=model)


@pytest.fixture(autouse=True)
def _api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERAGPT_API_KEY", "mg-live-test")
    monkeypatch.delenv("MERAGPT_BASE_URL", raising=False)
    monkeypatch.delenv("MERAGPT_MODEL", raising=False)
