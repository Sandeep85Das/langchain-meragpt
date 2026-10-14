"""`MeragptDecider`: Decider 1 as a LangChain Runnable."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from langchain_core.runnables import RunnableConfig, RunnableSerializable
from pydantic import ConfigDict, Field, PrivateAttr, SecretStr

from langchain_meragpt._client import (
    DEFAULT_MAX_RETRIES,
    DEFAULT_TIMEOUT,
    MeragptClient,
)


class MeragptDecider(RunnableSerializable[dict[str, Any], dict[str, Any]]):
    """Decider 1, a decision model: typed answers with calibrated probabilities.

    Invoke it with `{"state": ..., "questions": {...}}`; it returns the
    `answers` mapping, one entry per question. Fixed `questions` can be set on
    the instance, so a chain only has to supply the state.

    Args:
        api_key: Falls back to `MERAGPT_API_KEY`.
        base_url: Falls back to `MERAGPT_BASE_URL`, then `https://meragpt.com`.
        model: Falls back to `MERAGPT_MODEL`, then `decider-1`.
        questions: Default questions, used when the input has none.

    Example:
        ```python
        from langchain_meragpt import MeragptDecider, choice, yes_no

        decider = MeragptDecider()
        answers = decider.invoke({
            "state": {"ticket": "I was charged twice for order A-1042."},
            "questions": {
                "topic": choice("What is the ticket about?",
                                ["billing", "shipping", "account"]),
                "urgent": yes_no("Does this need a reply today?"),
            },
        })
        answers["topic"]["choice"]   # "billing"
        answers["urgent"]["noul"]    # probability of yes
        ```
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, populate_by_name=True)

    api_key: SecretStr | None = Field(default=None, repr=False)
    base_url: str | None = None
    model: str | None = None
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    questions: dict[str, Any] | None = None

    _client: MeragptClient = PrivateAttr()

    def model_post_init(self, __context: Any) -> None:
        self._client = MeragptClient(
            api_key=self.api_key.get_secret_value() if self.api_key else None,
            base_url=self.base_url,
            model=self.model,
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    @property
    def client(self) -> MeragptClient:
        """The underlying API client."""
        return self._client

    def _split(self, input: Any) -> tuple[Any, Mapping[str, Any]]:
        if isinstance(input, Mapping) and "state" in input:
            questions = input.get("questions") or self.questions
            state = input["state"]
        else:
            questions, state = self.questions, input
        if not questions:
            raise ValueError(
                "No questions: pass {'state': ..., 'questions': {...}} or set "
                "`questions=` on the MeragptDecider."
            )
        return state, questions

    def invoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Answer the questions; returns `answers`."""

        def run(inp: Any) -> dict[str, Any]:
            state, questions = self._split(inp)
            return self._client.decide(state, questions)["answers"]

        return self._call_with_config(run, input, config, run_type="tool")

    async def ainvoke(
        self, input: Any, config: RunnableConfig | None = None, **kwargs: Any
    ) -> dict[str, Any]:
        """Async version of `invoke`."""

        async def run(inp: Any) -> dict[str, Any]:
            state, questions = self._split(inp)
            return (await self._client.adecide(state, questions))["answers"]

        return await self._acall_with_config(run, input, config, run_type="tool")
