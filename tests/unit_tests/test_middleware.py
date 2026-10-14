from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool

from langchain_meragpt import (
    MeragptDecider,
    MeragptDecisionMiddleware,
    MeragptDecisionTool,
    MeragptRouterMiddleware,
    MeragptStopMiddleware,
    MeragptToolApprovalMiddleware,
    choice,
    conversation_state,
    yes_no,
)

from .conftest import FakeClient

ran: list[str] = []


@tool
def delete_files(path: str) -> str:
    """Delete files under a path."""
    ran.append(path)
    return f"deleted {path}"


class ToolFakeModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def calling_model(n_calls: int = 1) -> GenericFakeChatModel:
    msgs: list[AIMessage] = [
        AIMessage(
            content="",
            tool_calls=[{"name": "delete_files", "args": {"path": f"/tmp/{i}"}, "id": f"c{i}"}],
        )
        for i in range(n_calls)
    ]
    msgs.append(AIMessage(content="done"))
    return ToolFakeModel(messages=iter(msgs))


def user(text: str) -> dict[str, Any]:
    return {"messages": [{"role": "user", "content": text}]}


def test_conversation_state_shape() -> None:
    s = conversation_state(
        [
            HumanMessage("clean up"),
            AIMessage(
                content="", tool_calls=[{"name": "delete_files", "args": {"path": "/"}, "id": "1"}]
            ),
            ToolMessage("ok", tool_call_id="1", name="delete_files"),
        ]
    )
    assert s["conversation"][0] == {"role": "user", "text": "clean up"}
    assert s["conversation"][1]["tool_calls"] == [
        {"tool": "delete_files", "arguments": {"path": "/"}}
    ]
    assert s["conversation"][2]["tool"] == "delete_files"


def test_decision_middleware_stores_answers() -> None:
    fake = FakeClient(p=0.9)
    mw = MeragptDecisionMiddleware(
        {"escalate": yes_no("Human?"), "topic": choice("Topic?", ["billing", "tech"])}, client=fake
    )
    agent = create_agent(GenericFakeChatModel(messages=iter([AIMessage("hi")])), middleware=[mw])
    out = agent.invoke(user("I want a refund now"))
    assert out["meragpt_decisions"]["escalate"]["noul"] == 0.9
    assert out["meragpt_decisions"]["topic"]["choice"] == "billing"
    assert fake.calls[0][0]["conversation"][0]["text"] == "I want a refund now"


def test_decision_middleware_fails_open() -> None:
    mw = MeragptDecisionMiddleware({"x": yes_no("x?")}, client=FakeClient(fail=True))
    agent = create_agent(GenericFakeChatModel(messages=iter([AIMessage("hi")])), middleware=[mw])
    out = agent.invoke(user("hello"))
    assert "meragpt_decisions" not in out
    assert out["messages"][-1].content == "hi"


def test_decision_middleware_injects_summary() -> None:
    seen: list[str] = []

    class Spy(GenericFakeChatModel):
        def _generate(self, messages: Any, *a: Any, **k: Any) -> Any:
            seen.append(messages[0].content)
            return super()._generate(messages, *a, **k)

    mw = MeragptDecisionMiddleware(
        {"escalate": yes_no("Human?")}, inject_into_prompt=True, client=FakeClient(p=0.25)
    )
    agent = create_agent(
        Spy(messages=iter([AIMessage("ok")])), system_prompt="Be nice.", middleware=[mw]
    )
    agent.invoke(user("hello"))
    assert "escalate: p(yes)=0.25" in seen[0] and seen[0].startswith("Be nice.")


def test_tool_approval_blocks() -> None:
    ran.clear()
    guard = MeragptToolApprovalMiddleware(threshold=0.5, client=FakeClient(p=0.05))
    agent = create_agent(calling_model(), tools=[delete_files], middleware=[guard])
    out = agent.invoke(user("tidy my desktop"))
    assert ran == []
    blocked = [m for m in out["messages"] if isinstance(m, ToolMessage)][0]
    assert blocked.status == "error" and "blocked" in blocked.content


def test_tool_approval_allows_and_sends_context() -> None:
    ran.clear()
    fake = FakeClient(p=0.9)
    guard = MeragptToolApprovalMiddleware(client=fake)
    agent = create_agent(calling_model(), tools=[delete_files], middleware=[guard])
    agent.invoke(user("delete /tmp/0"))
    assert ran == ["/tmp/0"]
    state = fake.calls[0][0]
    assert state["user_request"] == "delete /tmp/0"
    assert state["tool"] == "delete_files" and state["arguments"] == {"path": "/tmp/0"}


def test_tool_approval_fail_open_and_closed() -> None:
    ran.clear()
    agent = create_agent(
        calling_model(),
        tools=[delete_files],
        middleware=[MeragptToolApprovalMiddleware(client=FakeClient(fail=True))],
    )
    agent.invoke(user("x"))
    assert ran == ["/tmp/0"]
    ran.clear()
    agent = create_agent(
        calling_model(),
        tools=[delete_files],
        middleware=[MeragptToolApprovalMiddleware(fail_closed=True, client=FakeClient(fail=True))],
    )
    agent.invoke(user("x"))
    assert ran == []


def test_tool_approval_only_targets() -> None:
    ran.clear()
    fake = FakeClient(p=0.01)
    agent = create_agent(
        calling_model(),
        tools=[delete_files],
        middleware=[MeragptToolApprovalMiddleware(tools=["send_email"], client=fake)],
    )
    agent.invoke(user("x"))
    assert ran == ["/tmp/0"] and fake.calls == []


def test_stop_middleware_ends_run() -> None:
    ran.clear()
    stop = MeragptStopMiddleware(min_tool_calls=2, threshold=0.8, client=FakeClient(p=0.9))
    agent = create_agent(calling_model(6), tools=[delete_files], middleware=[stop])
    out = agent.invoke(user("x"))
    assert len(ran) == 2
    assert out["messages"][-1].content.startswith("Stopped early: loop")


def test_stop_middleware_continues_when_unsure() -> None:
    ran.clear()
    stop = MeragptStopMiddleware(min_tool_calls=1, client=FakeClient(p=0.2))
    agent = create_agent(calling_model(3), tools=[delete_files], middleware=[stop])
    out = agent.invoke(user("x"))
    assert len(ran) == 3 and out["messages"][-1].content == "done"


def test_router_switches_model() -> None:
    small = GenericFakeChatModel(messages=iter([AIMessage("from small")]))
    big = GenericFakeChatModel(messages=iter([AIMessage("from big")]))
    router = MeragptRouterMiddleware(
        routes={"simple": ("Small talk", small), "complex": ("Hard problems", big)},
        client=FakeClient(p=0.9),  # FakeClient picks the first label
    )
    default = GenericFakeChatModel(messages=iter([AIMessage("from default")]))
    out = create_agent(default, middleware=[router]).invoke(user("hi"))
    assert out["messages"][-1].content == "from small"


def test_router_default_on_failure() -> None:
    small = GenericFakeChatModel(messages=iter([AIMessage("from small")]))
    big = GenericFakeChatModel(messages=iter([AIMessage("from big")]))
    router = MeragptRouterMiddleware(
        routes={"simple": ("Small talk", small), "complex": ("Hard", big)},
        default="complex",
        client=FakeClient(fail=True),
    )
    default = GenericFakeChatModel(messages=iter([AIMessage("from default")]))
    out = create_agent(default, middleware=[router]).invoke(user("hi"))
    assert out["messages"][-1].content == "from big"


async def test_async_paths() -> None:
    ran.clear()
    guard = MeragptToolApprovalMiddleware(client=FakeClient(p=0.05))
    decisions = MeragptDecisionMiddleware({"x": yes_no("x?")}, client=FakeClient(p=0.6))
    agent = create_agent(calling_model(), tools=[delete_files], middleware=[decisions, guard])
    out = await agent.ainvoke(user("x"))
    assert ran == [] and out["meragpt_decisions"]["x"]["noul"] == 0.6


def test_decider_runnable() -> None:
    d = MeragptDecider(questions={"u": yes_no("Urgent?")})
    d._client = FakeClient(p=0.7)
    assert d.invoke("server down")["u"]["noul"] == 0.7
    assert (
        d.invoke({"state": "x", "questions": {"t": choice("T?", ["a", "b"])}})["t"]["choice"] == "a"
    )
    with pytest.raises(ValueError):
        MeragptDecider().invoke("no questions")


def test_decider_hides_key() -> None:
    d = MeragptDecider(api_key="mg-live-secret")
    assert "mg-live-secret" not in repr(d)


def test_tool_formats_answers() -> None:
    t = MeragptDecisionTool()
    t._client = FakeClient(p=0.8)
    assert t.invoke({"state": "s", "question": "q?"}) == '{"p_yes": 0.8}'
    out = t.invoke({"state": "s", "question": "q?", "kind": "score", "options": ["low", "high"]})
    assert '"level": "low"' in out
    with pytest.raises(ValueError):
        t.invoke({"state": "s", "question": "q?", "kind": "choice"})
