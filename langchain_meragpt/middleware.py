"""Decider 1 middleware for LangChain agents.

Each middleware asks Decider 1 a typed question at one point in the agent loop
and acts on the probability it returns:

- `MeragptDecisionMiddleware`: answer your questions about the conversation
  once per run and keep the answers in state (triage, escalate, flag).
- `MeragptRouterMiddleware`: pick the model for each call from a set of routes.
- `MeragptToolApprovalMiddleware`: approve or block a tool call before it runs.
- `MeragptStopMiddleware`: stop an agent that is looping or has given up.

All of them fail open: if the meraGPT call fails, the agent carries on as if
the middleware were not there, and a warning is logged.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from langchain.agents.middleware import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    ToolCallRequest,
    hook_config,
)
from langchain.chat_models import BaseChatModel, init_chat_model
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from typing_extensions import NotRequired

from langchain_meragpt._client import MeragptAPIError, MeragptClient, _resolve_client
from langchain_meragpt.questions import choice, yes_no

if TYPE_CHECKING:
    from langgraph.runtime import Runtime
    from langgraph.types import Command

logger = logging.getLogger(__name__)

MAX_TEXT_CHARS = 2_000


def _clip(text: str, limit: int = MAX_TEXT_CHARS) -> str:
    return text if len(text) <= limit else text[: limit // 2] + " … " + text[-limit // 2 :]


def _role(m: AnyMessage) -> str:
    return {"human": "user", "ai": "assistant", "tool": "tool"}.get(m.type, m.type)


def conversation_state(messages: Sequence[AnyMessage], last: int = 20) -> dict[str, Any]:
    """The default state sent to Decider: the last `last` messages as turns.

    Each turn has a role and its text; an assistant turn that calls tools also
    lists the calls. Long texts are clipped in the middle.
    """
    turns: list[dict[str, Any]] = []
    for m in list(messages)[-last:]:
        turn: dict[str, Any] = {"role": _role(m), "text": _clip(m.text)}
        if isinstance(m, AIMessage) and m.tool_calls:
            turn["tool_calls"] = [{"tool": c["name"], "arguments": c["args"]} for c in m.tool_calls]
        if isinstance(m, ToolMessage):
            turn["tool"] = m.name
        turns.append(turn)
    return {"conversation": turns}


def _last_human(messages: Sequence[AnyMessage]) -> str | None:
    return next((m.text for m in reversed(messages) if isinstance(m, HumanMessage)), None)


# ---------------------------------------------------------------------------
# Decisions kept in state
# ---------------------------------------------------------------------------


class MeragptDecisionState(AgentState[Any]):
    """Agent state with the answers from `MeragptDecisionMiddleware`."""

    meragpt_decisions: NotRequired[dict[str, Any]]


class MeragptDecisionMiddleware(AgentMiddleware[MeragptDecisionState, Any, Any]):
    """Answer typed questions about the conversation before the agent runs.

    Once per invocation, before the first model call, the conversation is sent
    to Decider 1 with your `questions`. The answers are stored in agent state
    under `meragpt_decisions`, so tools, later middleware and the caller of
    `agent.invoke()` can branch on them:

    ```python
    {"escalate": {"type": "noul", "noul": 0.91},
     "topic": {"type": "choice", "choice": "billing", "probabilities": {...}}}
    ```

    Args:
        questions: Maps a name to a question built with `yes_no`, `choice` or
            `score`.
        state_builder: Builds the Decider state from the messages. Defaults to
            `conversation_state` (the last 20 turns).
        inject_into_prompt: If true, append a one-line summary of the answers
            to the system message on every model call. Off by default.
        client: Preconfigured `MeragptClient`. If omitted, one is created from
            `api_key` / `base_url` (or `MERAGPT_API_KEY` / `MERAGPT_BASE_URL`).

    Example:
        ```python
        from langchain.agents import create_agent
        from langchain_meragpt import MeragptDecisionMiddleware, choice, yes_no

        triage = MeragptDecisionMiddleware(questions={
            "topic": choice("What is the customer asking about?",
                            {"billing": "charges, refunds, invoices",
                             "technical": "bugs, errors, outages",
                             "account": "login, settings, cancellation"}),
            "escalate": yes_no("Is the customer asking for a human or threatening to leave?"),
        })
        agent = create_agent("openai:gpt-6-luna", tools=[...], middleware=[triage])
        result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})
        result["meragpt_decisions"]["escalate"]["noul"]
        ```
    """

    state_schema = MeragptDecisionState

    def __init__(
        self,
        questions: Mapping[str, Any],
        *,
        state_builder: Callable[[Sequence[AnyMessage]], Any] = conversation_state,
        inject_into_prompt: bool = False,
        client: MeragptClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        if not questions:
            raise ValueError("MeragptDecisionMiddleware needs at least one question.")
        self.questions = dict(questions)
        self.state_builder = state_builder
        self.inject_into_prompt = inject_into_prompt
        self.client = _resolve_client(client, api_key, base_url)

    def before_agent(
        self, state: MeragptDecisionState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Answer the questions over the conversation; fail open on errors."""
        try:
            data = self.client.decide(self.state_builder(state["messages"]), self.questions)
        except MeragptAPIError:
            logger.warning("meraGPT decision failed; continuing without it.", exc_info=True)
            return None
        return {"meragpt_decisions": data["answers"]}

    async def abefore_agent(
        self, state: MeragptDecisionState, runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of `before_agent`."""
        try:
            data = await self.client.adecide(self.state_builder(state["messages"]), self.questions)
        except MeragptAPIError:
            logger.warning("meraGPT decision failed; continuing without it.", exc_info=True)
            return None
        return {"meragpt_decisions": data["answers"]}

    @staticmethod
    def _summary(answers: Mapping[str, Any]) -> str:
        parts = []
        for name, a in answers.items():
            if a.get("type") == "noul":
                parts.append(f"{name}: p(yes)={a['noul']:.2f}")
            elif a.get("type") == "choice":
                parts.append(f"{name}: {a['choice']} ({a.get('confidence', 0):.2f})")
            else:
                parts.append(f"{name}: {a.get('score')}")
        return "Decisions about this conversation: " + "; ".join(parts) + "."

    def _inject(self, request: ModelRequest) -> ModelRequest:
        answers = request.state.get("meragpt_decisions") if request.state else None
        if not self.inject_into_prompt or not isinstance(answers, Mapping) or not answers:
            return request
        base = request.system_message.text if request.system_message else ""
        text = (base + "\n\n" if base else "") + self._summary(answers)
        return request.override(system_message=SystemMessage(content=text))

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Optionally add the answers to the system message."""
        return handler(self._inject(request))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of `wrap_model_call`."""
        return await handler(self._inject(request))


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


class MeragptRouterMiddleware(AgentMiddleware):
    """Choose the model for each call with a Decider `choice`.

    Args:
        routes: Maps a route label to `(description, model)`, where `model` is
            a chat model or a model string for `init_chat_model`. Up to 10.
        question: The routing question.
        default: Route to use when the decision fails or its confidence is
            below `min_confidence`. Defaults to keeping the agent's own model.
        min_confidence: Below this, use `default`.
        client: Preconfigured `MeragptClient` (see `MeragptDecisionMiddleware`).

    Example:
        ```python
        router = MeragptRouterMiddleware(routes={
            "simple": ("A short factual question or small talk", "openai:gpt-6-luna"),
            "complex": ("Multi-step reasoning, code or analysis", "anthropic:claude-opus-5-5"),
        })
        agent = create_agent("openai:gpt-6-luna", tools=[...], middleware=[router])
        ```
    """

    def __init__(
        self,
        routes: Mapping[str, tuple[str, str | BaseChatModel]],
        *,
        question: str = "Which model should handle the user's latest request?",
        default: str | None = None,
        min_confidence: float = 0.0,
        client: MeragptClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        if default is not None and default not in routes:
            raise ValueError(f"default {default!r} is not one of the routes.")
        self.question = choice(question, {k: d for k, (d, _) in routes.items()})
        self.models: dict[str, BaseChatModel] = {
            k: m if isinstance(m, BaseChatModel) else init_chat_model(m)
            for k, (_, m) in routes.items()
        }
        self.default = default
        self.min_confidence = min_confidence
        self.client = _resolve_client(client, api_key, base_url)

    def _state(self, request: ModelRequest) -> dict[str, Any] | None:
        text = _last_human(request.messages)
        return {"request": _clip(text)} if text else None

    def _route(self, request: ModelRequest, answers: Mapping[str, Any] | None) -> ModelRequest:
        label = None
        if answers:
            a = answers["route"]
            if a.get("confidence", 1.0) >= self.min_confidence:
                label = a["choice"]
        label = label or self.default
        return request.override(model=self.models[label]) if label else request

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Route the call to the chosen model."""
        state = self._state(request)
        answers = None
        if state is not None:
            try:
                answers = self.client.decide(state, {"route": self.question})["answers"]
            except MeragptAPIError:
                logger.warning("meraGPT routing failed; using the default.", exc_info=True)
        return handler(self._route(request, answers))

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        """Async version of `wrap_model_call`."""
        state = self._state(request)
        answers = None
        if state is not None:
            try:
                answers = (await self.client.adecide(state, {"route": self.question}))["answers"]
            except MeragptAPIError:
                logger.warning("meraGPT routing failed; using the default.", exc_info=True)
        return await handler(self._route(request, answers))


# ---------------------------------------------------------------------------
# Tool approval
# ---------------------------------------------------------------------------

# One narrow question separates far better than a composite "is it unsafe or
# unwanted?" (live probe, 2026-10-14: 0.02-0.20 for calls that do not fit the
# request, 0.86-0.91 for calls that do; the composite question scored 0.41-0.53
# on both).
DEFAULT_APPROVE_QUESTION = "Does this tool call do what the user asked for?"


class MeragptToolApprovalMiddleware(AgentMiddleware):
    """Approve or block a tool call with a Decider yes/no before it runs.

    Decider sees the user's latest request, the tool's name and description,
    and the arguments, and answers `question`, where "yes" approves the call.
    When p(yes) is below `threshold`, the tool does not run and the model
    receives an error `ToolMessage` saying why, so it can try another way or
    ask the user.

    Keep the question narrow. "Does this tool call do what the user asked
    for?" (the default) works well; a composite such as "is this unsafe or
    unwanted?" separates poorly.

    Args:
        tools: Tools (or names) to check. Defaults to every tool.
        question: The yes/no question; "yes" approves the call.
        threshold: Run the tool only when p(yes) is at least this.
        fail_closed: If true, block when the meraGPT call fails. Defaults to
            false (fail open).
        client: Preconfigured `MeragptClient` (see `MeragptDecisionMiddleware`).

    Example:
        ```python
        guard = MeragptToolApprovalMiddleware(tools=["run_shell", "send_email"])
        agent = create_agent("openai:gpt-6-luna", tools=[run_shell, send_email],
                             middleware=[guard])
        ```
    """

    def __init__(
        self,
        *,
        tools: Sequence[str | BaseTool] | None = None,
        question: str = DEFAULT_APPROVE_QUESTION,
        threshold: float = 0.5,
        fail_closed: bool = False,
        client: MeragptClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self.tool_names = (
            None if tools is None else {t if isinstance(t, str) else t.name for t in tools}
        )
        self.question = yes_no(question)
        self.threshold = threshold
        self.fail_closed = fail_closed
        self.client = _resolve_client(client, api_key, base_url)

    def _targets(self, request: ToolCallRequest) -> bool:
        return self.tool_names is None or request.tool_call["name"] in self.tool_names

    @staticmethod
    def _state(request: ToolCallRequest) -> dict[str, Any]:
        messages = (
            (request.state or {}).get("messages", []) if isinstance(request.state, Mapping) else []
        )
        state: dict[str, Any] = {
            "user_request": _clip(_last_human(messages) or ""),
            "tool": request.tool_call["name"],
            "arguments": request.tool_call["args"],
        }
        if request.tool is not None and request.tool.description:
            state["tool_description"] = _clip(request.tool.description, 500)
        return state

    def _verdict(self, answers: Mapping[str, Any] | None) -> float | None:
        if answers is None:
            return 0.0 if self.fail_closed else None
        return float(answers["approve"]["noul"])

    def _blocked(self, request: ToolCallRequest, p: float) -> ToolMessage:
        return ToolMessage(
            content=(
                f'Tool call blocked: p={p:.2f} for "{self.question["instructions"]}" '
                "Do not retry it unchanged; choose another approach or ask the user."
            ),
            tool_call_id=request.tool_call["id"],
            name=request.tool_call["name"],
            status="error",
        )

    def wrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command[Any]],
    ) -> ToolMessage | Command[Any]:
        """Run the tool only if Decider approves it."""
        if not self._targets(request):
            return handler(request)
        try:
            answers = self.client.decide(self._state(request), {"approve": self.question})[
                "answers"
            ]
        except MeragptAPIError:
            logger.warning("meraGPT tool approval failed.", exc_info=True)
            answers = None
        p = self._verdict(answers)
        if p is not None and p < self.threshold:
            return self._blocked(request, p)
        return handler(request)

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], Awaitable[ToolMessage | Command[Any]]],
    ) -> ToolMessage | Command[Any]:
        """Async version of `wrap_tool_call`."""
        if not self._targets(request):
            return await handler(request)
        try:
            answers = (await self.client.adecide(self._state(request), {"approve": self.question}))[
                "answers"
            ]
        except MeragptAPIError:
            logger.warning("meraGPT tool approval failed.", exc_info=True)
            answers = None
        p = self._verdict(answers)
        if p is not None and p < self.threshold:
            return self._blocked(request, p)
        return await handler(request)


# ---------------------------------------------------------------------------
# Stop / continue
# ---------------------------------------------------------------------------

DEFAULT_STOP_QUESTIONS: dict[str, dict[str, Any]] = {
    "loop": yes_no("Is the agent stuck repeating the same failing step?"),
    "gave_up": yes_no("Has the agent given up on the task?"),
}


class MeragptStopMiddleware(AgentMiddleware):
    """Stop an agent that is looping or has given up.

    After each model call that requests more tool calls, the recent trace is
    sent to Decider 1. If any answer's p(yes) reaches `threshold`, the agent
    jumps to the end with a short message saying why, instead of spending more
    calls.

    Args:
        questions: Yes/no questions where "yes" means stop. Defaults to
            `loop` and `gave_up`.
        threshold: Stop when any p(yes) is at least this.
        min_tool_calls: Only check once the run has made this many tool calls.
        last: How many recent messages to send.
        client: Preconfigured `MeragptClient` (see `MeragptDecisionMiddleware`).
    """

    def __init__(
        self,
        *,
        questions: Mapping[str, Any] | None = None,
        threshold: float = 0.7,
        min_tool_calls: int = 4,
        last: int = 20,
        client: MeragptClient | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self.questions = dict(questions or DEFAULT_STOP_QUESTIONS)
        self.threshold = threshold
        self.min_tool_calls = min_tool_calls
        self.last = last
        self.client = _resolve_client(client, api_key, base_url)

    def _due(self, messages: Sequence[AnyMessage]) -> bool:
        if not messages or not isinstance(messages[-1], AIMessage):
            return False
        if not messages[-1].tool_calls:
            return False  # the agent is finishing anyway
        return sum(isinstance(m, ToolMessage) for m in messages) >= self.min_tool_calls

    def _stop(self, answers: Mapping[str, Any]) -> dict[str, Any] | None:
        hits = {
            k: float(a["noul"])
            for k, a in answers.items()
            if a.get("type") == "noul" and float(a["noul"]) >= self.threshold
        }
        if not hits:
            return None
        reasons = ", ".join(f"{k} (p={p:.2f})" for k, p in hits.items())
        return {
            "jump_to": "end",
            "messages": [AIMessage(content=f"Stopped early: {reasons}.")],
        }

    @hook_config(can_jump_to=["end"])
    def after_model(self, state: AgentState[Any], runtime: Runtime[Any]) -> dict[str, Any] | None:
        """Check the trace; jump to the end if the agent should stop."""
        messages = state["messages"]
        if not self._due(messages):
            return None
        try:
            answers = self.client.decide(conversation_state(messages, self.last), self.questions)[
                "answers"
            ]
        except MeragptAPIError:
            logger.warning("meraGPT stop check failed; continuing.", exc_info=True)
            return None
        return self._stop(answers)

    @hook_config(can_jump_to=["end"])
    async def aafter_model(
        self, state: AgentState[Any], runtime: Runtime[Any]
    ) -> dict[str, Any] | None:
        """Async version of `after_model`."""
        messages = state["messages"]
        if not self._due(messages):
            return None
        try:
            answers = (
                await self.client.adecide(conversation_state(messages, self.last), self.questions)
            )["answers"]
        except MeragptAPIError:
            logger.warning("meraGPT stop check failed; continuing.", exc_info=True)
            return None
        return self._stop(answers)
