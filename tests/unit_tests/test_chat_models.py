"""Unit tests for ChatMeragpt, with the HTTP layer replaced."""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_meragpt import ChatMeragpt, MeragptAPIError

RESPONSE = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 1788000000,
    "model": "restyler-1",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Our platform helps teams work together."},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 58, "completion_tokens": 49, "total_tokens": 107},
    "meragpt": {"blocks_rewritten": 1, "blocks_total": 1, "cost_nano_usd": 9150},
}

STREAM = [
    {"id": "c1", "choices": [{"index": 0, "delta": {"role": "assistant", "content": "First."}}]},
    {"id": "c1", "choices": [{"index": 0, "delta": {"content": "\n\nSecond."}}]},
    {
        "id": "c1",
        "model": "restyler-1",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 6, "total_tokens": 16},
        "meragpt": {"blocks_rewritten": 2, "blocks_total": 2},
    },
]


def _sse(frames: list[dict[str, Any]]) -> bytes:
    return b"".join(f"data: {json.dumps(f)}\n\n".encode() for f in frames) + b"data: [DONE]\n\n"


class Recorder:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self.responses = responses
        self.bodies: list[dict[str, Any]] = []
        self.headers: list[httpx.Headers] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.bodies.append(json.loads(request.content))
        self.headers.append(request.headers)
        return self.responses.pop(0)


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Route every httpx call through a MockTransport driven by a Recorder."""

    def install(responses: list[httpx.Response]) -> Recorder:
        rec = Recorder(responses)
        mock = httpx.MockTransport(rec)
        real_client, real_async = httpx.Client, httpx.AsyncClient

        def post(url: str, **kw: Any) -> httpx.Response:
            with real_client(transport=mock) as c:
                return c.post(url, **{k: v for k, v in kw.items() if k != "timeout"})

        class StreamCtx:
            def __init__(self, method: str, url: str, **kw: Any) -> None:
                self.c = real_client(transport=mock)
                kw.pop("timeout", None)
                self.cm = self.c.stream(method, url, **kw)

            def __enter__(self) -> httpx.Response:
                return self.cm.__enter__()

            def __exit__(self, *a: Any) -> None:
                self.cm.__exit__(*a)
                self.c.close()

        def async_client(**kw: Any) -> httpx.AsyncClient:
            kw.pop("timeout", None)
            return real_async(transport=mock, **kw)

        monkeypatch.setattr(httpx, "post", post)
        monkeypatch.setattr(httpx, "stream", StreamCtx)
        monkeypatch.setattr(httpx, "AsyncClient", async_client)
        monkeypatch.setattr("time.sleep", lambda s: None)
        return rec

    return install


def test_defaults() -> None:
    llm = ChatMeragpt()
    assert llm.model_name == "restyler-1"
    assert llm.base_url == "https://meragpt.com"
    assert llm._url == "https://meragpt.com/v1/chat/completions"


def test_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MERAGPT_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MERAGPT_API_KEY"):
        ChatMeragpt()


def test_base_url_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MERAGPT_BASE_URL", "https://proxy.example/")
    assert ChatMeragpt()._url == "https://proxy.example/v1/chat/completions"


def test_invoke_sends_only_last_user_message(transport: Any) -> None:
    rec = transport([httpx.Response(200, json=RESPONSE)])
    llm = ChatMeragpt()
    out = llm.invoke(
        [
            SystemMessage("be brief"),
            HumanMessage("old draft"),
            AIMessage("old reply"),
            HumanMessage("The utilization of our platform facilitates synergy."),
        ]
    )
    assert rec.bodies == [
        {
            "model": "restyler-1",
            "messages": [
                {"role": "user", "content": "The utilization of our platform facilitates synergy."}
            ],
        }
    ]
    assert rec.headers[0]["authorization"] == "Bearer mg-live-test"
    assert out.content == "Our platform helps teams work together."
    assert out.usage_metadata == {"input_tokens": 58, "output_tokens": 49, "total_tokens": 107}
    assert out.response_metadata["meragpt"]["blocks_rewritten"] == 1
    assert out.response_metadata["model_provider"] == "meragpt"
    assert out.response_metadata["finish_reason"] == "stop"


def test_invoke_string(transport: Any) -> None:
    rec = transport([httpx.Response(200, json=RESPONSE)])
    ChatMeragpt().invoke("draft text")
    assert rec.bodies[0]["messages"] == [{"role": "user", "content": "draft text"}]


def test_no_user_message() -> None:
    with pytest.raises(ValueError, match="last user message"):
        ChatMeragpt().invoke([SystemMessage("only a system message")])


def test_retry_then_success(transport: Any) -> None:
    rec = transport(
        [
            httpx.Response(429, headers={"retry-after": "1"}, json={"error": {"code": "x"}}),
            httpx.Response(200, json=RESPONSE),
        ]
    )
    assert ChatMeragpt().invoke("draft").content
    assert len(rec.bodies) == 2


def test_error_is_raised(transport: Any) -> None:
    transport(
        [
            httpx.Response(
                400,
                json={
                    "error": {
                        "message": "too long",
                        "type": "invalid_request_error",
                        "code": "input_too_large",
                    }
                },
            )
        ]
    )
    with pytest.raises(MeragptAPIError) as e:
        ChatMeragpt().invoke("draft")
    assert e.value.status == 400 and e.value.code == "input_too_large"


def test_stream(transport: Any) -> None:
    rec = transport([httpx.Response(200, content=_sse(STREAM))])
    chunks = list(ChatMeragpt().stream("draft"))
    assert rec.bodies[0]["stream"] is True
    assert "".join(c.text for c in chunks) == "First.\n\nSecond."
    total = chunks[0]
    for c in chunks[1:]:
        total += c
    assert total.usage_metadata == {"input_tokens": 10, "output_tokens": 6, "total_tokens": 16}
    assert total.response_metadata["meragpt"]["blocks_total"] == 2


def test_stream_error_frame(transport: Any) -> None:
    frames = [STREAM[0], {"error": {"message": "capacity", "code": "capacity_saturated"}}]
    transport([httpx.Response(200, content=_sse(frames))])
    with pytest.raises(MeragptAPIError, match="capacity"):
        list(ChatMeragpt().stream("draft"))


async def test_ainvoke(transport: Any) -> None:
    rec = transport([httpx.Response(200, json=RESPONSE)])
    out = await ChatMeragpt().ainvoke("draft")
    assert out.content == "Our platform helps teams work together."
    assert rec.bodies[0]["model"] == "restyler-1"


async def test_astream(transport: Any) -> None:
    transport([httpx.Response(200, content=_sse(STREAM))])
    text = "".join([c.text async for c in ChatMeragpt().astream("draft")])
    assert text == "First.\n\nSecond."


def test_no_tool_calling() -> None:
    with pytest.raises(NotImplementedError):
        ChatMeragpt().bind_tools([])
