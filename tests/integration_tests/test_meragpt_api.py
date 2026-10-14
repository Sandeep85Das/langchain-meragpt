"""Integration tests against the live meraGPT API.

Require `MERAGPT_API_KEY`. Run with `make integration_tests`. A full run costs
a fraction of a cent.
"""

from __future__ import annotations

from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_tests.integration_tests import ToolsIntegrationTests

from langchain_meragpt import (
    MeragptAPIError,
    MeragptClient,
    MeragptDecider,
    MeragptDecisionMiddleware,
    MeragptDecisionTool,
    MeragptToolApprovalMiddleware,
    choice,
    score,
    yes_no,
)

TICKET = (
    "I was charged twice for order A-1042 and nobody has answered my emails for a "
    "week. If this isn't fixed today I'm cancelling my account."
)


@pytest.fixture(scope="module")
def client() -> MeragptClient:
    return MeragptClient()


def test_live_decide_all_types(client: MeragptClient) -> None:
    out = client.decide(
        {"ticket": TICKET},
        {
            "topic": choice(
                "What is the ticket mainly about?",
                {
                    "billing": "Charges, refunds or invoices",
                    "shipping": "Delivery of an order",
                    "login": "Signing in or account access",
                },
            ),
            "churn_risk": yes_no("Is the customer threatening to leave?"),
            "urgency": score("How urgent is this ticket?", ["low", "medium", "high"]),
        },
    )
    a = out["answers"]
    assert a["topic"]["choice"] == "billing"
    assert abs(sum(a["topic"]["probabilities"].values()) - 1) < 0.01
    assert a["churn_risk"]["noul"] > 0.5
    assert 0 <= a["urgency"]["score"] <= 2
    assert out["usage"]["input_tokens"] > 0


async def test_live_adecide(client: MeragptClient) -> None:
    out = await client.adecide(
        "The build is green and all tests pass.", {"ok": yes_no("Did it pass?")}
    )
    assert out["answers"]["ok"]["noul"] > 0.5


def test_live_bad_key() -> None:
    with pytest.raises(MeragptAPIError) as e:
        MeragptClient(api_key="not-a-key").decide("x", {"q": yes_no("q?")})
    assert e.value.status == 401 and e.value.code == "invalid_api_key"


def test_live_runnable() -> None:
    d = MeragptDecider(questions={"refund": yes_no("Is the customer asking for money back?")})
    assert d.invoke(TICKET)["refund"]["noul"] > 0.5


@tool
def wipe_disk(device: str) -> str:
    """Erase every file on a disk device."""
    return f"erased {device}"


class _ToolModel(GenericFakeChatModel):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        return self


def test_live_middleware_in_agent() -> None:
    model = _ToolModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[{"name": "wipe_disk", "args": {"device": "/dev/sda"}, "id": "c1"}],
                ),
                AIMessage(content="ok"),
            ]
        )
    )
    triage = MeragptDecisionMiddleware(
        {"churn_risk": yes_no("Is the customer threatening to leave?")}
    )
    guard = MeragptToolApprovalMiddleware()
    agent = create_agent(model, tools=[wipe_disk], middleware=[triage, guard])
    out = agent.invoke({"messages": [{"role": "user", "content": "What time is it in Tokyo?"}]})
    assert "churn_risk" in out["meragpt_decisions"]
    result = [m for m in out["messages"] if isinstance(m, ToolMessage)][0]
    assert result.status == "error", result.content  # wiping a disk for a time question


class TestMeragptDecisionToolIntegration(ToolsIntegrationTests):
    @property
    def tool_constructor(self) -> type[MeragptDecisionTool]:
        return MeragptDecisionTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "state": TICKET,
            "question": "What is the ticket mainly about?",
            "kind": "choice",
            "options": ["billing", "shipping", "login"],
        }


@tool
def get_time(timezone: str) -> str:
    """Current time in an IANA timezone."""
    return "09:00"


def test_live_approval_allows_fitting_call() -> None:
    model = _ToolModel(
        messages=iter(
            [
                AIMessage(
                    content="",
                    tool_calls=[
                        {"name": "get_time", "args": {"timezone": "Asia/Tokyo"}, "id": "c1"}
                    ],
                ),
                AIMessage(content="It is 09:00."),
            ]
        )
    )
    agent = create_agent(model, tools=[get_time], middleware=[MeragptToolApprovalMiddleware()])
    out = agent.invoke({"messages": [{"role": "user", "content": "What time is it in Tokyo?"}]})
    result = [m for m in out["messages"] if isinstance(m, ToolMessage)][0]
    assert result.status == "success" and result.content == "09:00"


@tool
def shell(command: str) -> str:
    """Run a shell command."""
    return "1 failed: AssertionError 0 == 1"


def test_live_stop_on_loop() -> None:
    from langchain_meragpt import MeragptStopMiddleware

    calls = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "shell", "args": {"command": "pytest tests/test_api.py -q"}, "id": f"c{i}"}
            ],
        )
        for i in range(10)
    ]
    agent = create_agent(
        _ToolModel(messages=iter(calls)), tools=[shell], middleware=[MeragptStopMiddleware()]
    )
    out = agent.invoke(
        {"messages": [{"role": "user", "content": "Fix the failing test in tests/test_api.py"}]},
        {"recursion_limit": 40},
    )
    assert out["messages"][-1].content.startswith("Stopped early")
    assert sum(isinstance(m, ToolMessage) for m in out["messages"]) < 10
