"""`MeragptDecisionTool`: let an agent ask Decider 1 a typed question."""

from __future__ import annotations

import json
from typing import Any, Literal

from langchain_core.callbacks import (
    AsyncCallbackManagerForToolRun,
    CallbackManagerForToolRun,
)
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, SecretStr

from langchain_meragpt._client import MeragptClient
from langchain_meragpt.questions import choice, score, yes_no


class DecisionInput(BaseModel):
    """Input for `MeragptDecisionTool`."""

    state: str = Field(description="The text to decide about.")
    question: str = Field(description="The question to answer about the text.")
    kind: Literal["yes_no", "choice", "score"] = Field(
        default="yes_no",
        description=(
            "yes_no for a probability of yes; choice to pick one of `options`; "
            "score to place the text on the ordered scale in `options`, lowest first."
        ),
    )
    options: list[str] | None = Field(
        default=None,
        description="Up to 10 options (choice) or levels, lowest first (score).",
    )


class MeragptDecisionTool(BaseTool):
    """Answer a yes/no, pick-one or scale question about a text with Decider 1.

    The result is JSON: for yes_no, `{"p_yes": 0.83}`; for choice,
    `{"choice": "billing", "probabilities": {...}}`; for score,
    `{"score": 1.4, "level": "medium", "probabilities": {...}}`.

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_meragpt import MeragptDecisionTool

        agent = create_agent("openai:gpt-6-luna", tools=[MeragptDecisionTool()])
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = "meragpt_decide"
    description: str = (
        "Answer a yes/no, pick-one or scale question about a piece of text, with "
        "calibrated probabilities. Use it for classification, routing, triage and "
        "approval decisions."
    )
    args_schema: type[BaseModel] = DecisionInput

    api_key: SecretStr | None = Field(default=None, repr=False)
    base_url: str | None = None
    model: str | None = None

    _client: MeragptClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = MeragptClient(
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            base_url=self.base_url,
            model=self.model,
        )

    @staticmethod
    def _question(question: str, kind: str, options: list[str] | None) -> dict[str, Any]:
        if kind == "yes_no":
            return yes_no(question)
        if not options:
            raise ValueError(f"A {kind} question needs `options`.")
        return choice(question, options) if kind == "choice" else score(question, options)

    @staticmethod
    def _format(answer: dict[str, Any]) -> str:
        if answer["type"] == "noul":
            return json.dumps({"p_yes": answer["noul"]})
        if answer["type"] == "choice":
            return json.dumps(
                {"choice": answer["choice"], "probabilities": answer["probabilities"]}
            )
        legend = answer.get("legend") or {}
        probs = answer.get("probabilities") or {}
        top = max(probs, key=lambda k: probs[k]) if probs else None
        return json.dumps(
            {
                "score": answer["score"],
                "level": legend.get(top, top),
                "probabilities": {legend.get(k, k): v for k, v in probs.items()},
            }
        )

    def _run(
        self,
        state: str,
        question: str,
        kind: str = "yes_no",
        options: list[str] | None = None,
        run_manager: CallbackManagerForToolRun | None = None,
    ) -> str:
        q = self._question(question, kind, options)
        answers = self._client.decide(state, {"answer": q})["answers"]
        return self._format(answers["answer"])

    async def _arun(
        self,
        state: str,
        question: str,
        kind: str = "yes_no",
        options: list[str] | None = None,
        run_manager: AsyncCallbackManagerForToolRun | None = None,
    ) -> str:
        q = self._question(question, kind, options)
        answers = (await self._client.adecide(state, {"answer": q}))["answers"]
        return self._format(answers["answer"])
