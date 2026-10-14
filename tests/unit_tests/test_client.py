from __future__ import annotations

import json
from typing import Any

import httpx
import pytest

from langchain_meragpt import MeragptAPIError, MeragptClient, choice, probability, score, yes_no
from langchain_meragpt import _client as client_module

OK = {
    "id": "req_1",
    "object": "systemone",
    "model": "decider-1",
    "answers": {"urgent": {"type": "noul", "noul": 0.81}},
    "usage": {"input_tokens": 40, "output_tokens": 0, "cost_usd": "$0.000002"},
}


def install(
    monkeypatch: pytest.MonkeyPatch, responses: list[httpx.Response]
) -> list[httpx.Request]:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return responses[min(len(seen) - 1, len(responses) - 1)]

    transport = httpx.MockTransport(handler)
    real_client, real_async = httpx.Client, httpx.AsyncClient

    def fake_post(url: str, **kw: Any) -> httpx.Response:
        timeout = kw.pop("timeout", None)
        with real_client(transport=transport, timeout=timeout) as c:
            return c.post(url, **kw)

    monkeypatch.setattr(client_module.httpx, "post", fake_post)
    monkeypatch.setattr(
        client_module.httpx, "AsyncClient", lambda **kw: real_async(transport=transport, **kw)
    )
    monkeypatch.setattr(client_module.time, "sleep", lambda s: None)

    async def no_sleep(s: float) -> None:
        return None

    monkeypatch.setattr(client_module.asyncio, "sleep", no_sleep)
    return seen


def test_requires_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MERAGPT_API_KEY")
    with pytest.raises(ValueError, match="MERAGPT_API_KEY"):
        MeragptClient()


def test_env_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERAGPT_BASE_URL", "https://gateway.example.com/")
    monkeypatch.setenv("MERAGPT_MODEL", "sd-1")
    c = MeragptClient()
    assert c.base_url == "https://gateway.example.com"
    assert c.model == "sd-1"


def test_decide_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = install(monkeypatch, [httpx.Response(200, json=OK)])
    out = MeragptClient().decide({"ticket": "hi"}, {"urgent": yes_no("Urgent?")})
    assert out["answers"]["urgent"]["noul"] == 0.81
    req = seen[0]
    assert str(req.url) == "https://meragpt.com/v1/systemone"
    assert req.headers["authorization"] == "Bearer mg-live-test"
    body = json.loads(req.content)
    assert body == {
        "model": "decider-1",
        "state": {"ticket": "hi"},
        "questions": {"urgent": {"type": "noul", "instructions": "Urgent?"}},
    }


def test_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = install(
        monkeypatch,
        [
            httpx.Response(
                429, headers={"Retry-After": "1"}, json={"error": {"code": "capacity_saturated"}}
            ),
            httpx.Response(200, json=OK),
        ],
    )
    MeragptClient().decide("x", {"urgent": yes_no("Urgent?")})
    assert len(seen) == 2


def test_error_envelope(monkeypatch: pytest.MonkeyPatch) -> None:
    install(
        monkeypatch,
        [
            httpx.Response(
                402,
                json={
                    "error": {
                        "message": "Your credit balance is empty.",
                        "type": "insufficient_quota",
                        "code": "insufficient_credit",
                    }
                },
            )
        ],
    )
    with pytest.raises(MeragptAPIError) as e:
        MeragptClient().decide("x", {"urgent": yes_no("Urgent?")})
    assert e.value.status == 402
    assert e.value.code == "insufficient_credit"


def test_retries_exhausted(monkeypatch: pytest.MonkeyPatch) -> None:
    seen = install(monkeypatch, [httpx.Response(503, json={"error": {"code": "no_capacity"}})])
    with pytest.raises(MeragptAPIError):
        MeragptClient(max_retries=2).decide("x", {"u": yes_no("u?")})
    assert len(seen) == 3


async def test_adecide(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, [httpx.Response(200, json=OK)])
    out = await MeragptClient().adecide("x", {"urgent": yes_no("Urgent?")})
    assert out["answers"]["urgent"]["type"] == "noul"


def test_question_builders() -> None:
    assert choice("Topic?", ["a", "b"]) == {
        "type": "choice",
        "instructions": "Topic?",
        "criteria": {"a": None, "b": None},
    }
    assert score("Urgency?", ["low", "high"])["criteria"] == ["low", "high"]
    with pytest.raises(ValueError):
        choice("Too many?", [str(i) for i in range(11)])
    with pytest.raises(ValueError):
        MeragptClient().decide("x", {})


def test_probability() -> None:
    assert probability({"type": "noul", "noul": 0.3}) == 0.3
    a = {"type": "choice", "choice": "b", "confidence": 0.7, "probabilities": {"a": 0.3, "b": 0.7}}
    assert probability(a) == 0.7
    assert probability(a, "a") == 0.3
