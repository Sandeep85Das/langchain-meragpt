"""Chat model for meraGPT's text models over `POST /v1/chat/completions`.

Restyler 1 (`restyler-1`) rewrites machine-written prose. The chat endpoint is a
compatibility layer over a single-task model (https://meragpt.com/docs/chat-completions):
only the last user message is used, sampling parameters are ignored, and tools,
`n`, `logprobs` and `response_format` are not supported. Decider 1 is not served
there; use `MeragptDecider` for it.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Iterator
from typing import Any

import httpx
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models import BaseChatModel, LangSmithParams
from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
)
from langchain_core.messages.ai import UsageMetadata
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import from_env, secret_from_env
from pydantic import ConfigDict, Field, SecretStr, model_validator
from typing_extensions import Self

from langchain_meragpt._client import (
    DEFAULT_BASE_URL,
    RETRY_STATUSES,
    MeragptAPIError,
    _error_from,
    _retry_wait,
)

DEFAULT_CHAT_MODEL = "restyler-1"


def _last_user_text(messages: list[BaseMessage]) -> str:
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            text = message.text
            if text.strip():
                return text
            break
    raise ValueError(
        "meraGPT chat models read only the last user message, and none with text was "
        "found. Pass the text to restyle as a user message."
    )


def _usage(data: dict[str, Any] | None) -> UsageMetadata | None:
    if not data:
        return None
    prompt = int(data.get("prompt_tokens") or 0)
    completion = int(data.get("completion_tokens") or 0)
    return UsageMetadata(
        input_tokens=prompt,
        output_tokens=completion,
        total_tokens=int(data.get("total_tokens") or prompt + completion),
    )


class ChatMeragpt(BaseChatModel):
    """meraGPT Restyler 1 as a LangChain chat model.

    Restyler 1 rewrites machine-written prose so it reads the way a person writes,
    and is trained to keep facts, names, numbers, links and formatting. Send the
    text to rewrite as the user message; the reply is the rewritten text.

    Setup:
        ```bash
        pip install -U langchain-meragpt
        export MERAGPT_API_KEY="mg-live-..."
        ```

    Key init args:
        model: Model id. Defaults to `restyler-1`.
        api_key: meraGPT API key. Read from `MERAGPT_API_KEY` if not passed.
        base_url: API base URL. Read from `MERAGPT_BASE_URL`, default
            `https://meragpt.com`.
        timeout: Request timeout in seconds.
        max_retries: Retries on 429 and 5xx capacity answers, honouring
            `Retry-After`.

    Instantiate:
        ```python
        from langchain_meragpt import ChatMeragpt

        restyler = ChatMeragpt()
        ```

    Invoke:
        ```python
        restyler.invoke(
            "The utilization of our platform facilitates the optimization of "
            "workflow efficiency across organizational boundaries."
        ).content
        ```

    Only the last user message is sent. System messages and earlier turns are
    dropped, since the model reads one document at a time. Sampling parameters are
    not offered because decoding is fixed per model. Streaming yields one
    paragraph per chunk, not one token. Up to 6,000 characters per request.
    """

    model_name: str = Field(default=DEFAULT_CHAT_MODEL, alias="model")
    """Model id."""
    api_key: SecretStr | None = Field(
        default_factory=secret_from_env("MERAGPT_API_KEY", default=None)
    )
    """meraGPT API key, read from `MERAGPT_API_KEY` if not provided."""
    base_url: str = Field(default_factory=from_env("MERAGPT_BASE_URL", default=DEFAULT_BASE_URL))
    """API base URL, read from `MERAGPT_BASE_URL` if not provided."""
    timeout: float = 180.0
    """Request timeout in seconds. The API allows up to 180 seconds per block."""
    max_retries: int = 2
    """Retries on 429 and 5xx capacity answers."""

    model_config = ConfigDict(populate_by_name=True)

    @model_validator(mode="after")
    def _check_key(self) -> Self:
        if not (self.api_key and self.api_key.get_secret_value()):
            raise ValueError(
                "No meraGPT API key found. Pass `api_key=...` or set the "
                "MERAGPT_API_KEY environment variable. Create a key at "
                "https://meragpt.com/dashboard/keys"
            )
        return self

    @property
    def _llm_type(self) -> str:
        return "chat-meragpt"

    @property
    def _identifying_params(self) -> dict[str, Any]:
        return {"model_name": self.model_name}

    @property
    def lc_secrets(self) -> dict[str, str]:
        return {"api_key": "MERAGPT_API_KEY"}

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        return LangSmithParams(
            ls_provider="meragpt",
            ls_model_name=self.model_name,
            ls_model_type="chat",
        )

    # -- request helpers ------------------------------------------------------

    @property
    def _url(self) -> str:
        return f"{self.base_url.rstrip('/')}/v1/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        assert self.api_key is not None
        return {
            "Authorization": f"Bearer {self.api_key.get_secret_value()}",
            "Content-Type": "application/json",
            "User-Agent": "langchain-meragpt",
        }

    def _body(self, messages: list[BaseMessage], stream: bool) -> dict[str, Any]:
        body: dict[str, Any] = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": _last_user_text(messages)}],
        }
        if stream:
            body["stream"] = True
        return body

    def _result(self, data: dict[str, Any]) -> ChatResult:
        choices = data.get("choices") or []
        if not choices:
            raise MeragptAPIError(f"meraGPT returned no choices: {data!r}")
        choice = choices[0]
        content = (choice.get("message") or {}).get("content") or ""
        metadata: dict[str, Any] = {
            "model_name": data.get("model", self.model_name),
            "model_provider": "meragpt",
            "finish_reason": choice.get("finish_reason"),
            "id": data.get("id"),
        }
        if "meragpt" in data:
            metadata["meragpt"] = data["meragpt"]
        message = AIMessage(
            content=content,
            id=data.get("id"),
            response_metadata=metadata,
            usage_metadata=_usage(data.get("usage")),
        )
        return ChatResult(generations=[ChatGeneration(message=message)])

    def _chunk(self, data: dict[str, Any]) -> ChatGenerationChunk | None:
        if data.get("error"):
            err = data["error"]
            raise MeragptAPIError(
                f"meraGPT stream error: {err.get('message') or err}",
                code=err.get("code"),
                type=err.get("type"),
            )
        choices = data.get("choices") or []
        text = ""
        finish_reason = None
        if choices:
            text = (choices[0].get("delta") or {}).get("content") or ""
            finish_reason = choices[0].get("finish_reason")
        metadata: dict[str, Any] = {"model_provider": "meragpt"}
        if finish_reason:
            metadata["finish_reason"] = finish_reason
            metadata["model_name"] = data.get("model", self.model_name)
        if "meragpt" in data:
            metadata["meragpt"] = data["meragpt"]
        usage = _usage(data.get("usage"))
        if not text and usage is None and len(metadata) == 1:
            return None
        return ChatGenerationChunk(
            message=AIMessageChunk(
                content=text,
                id=data.get("id"),
                response_metadata=metadata,
                usage_metadata=usage,
            )
        )

    @staticmethod
    def _sse_data(line: str) -> dict[str, Any] | None:
        if not line.startswith("data:"):
            return None
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            return None
        return json.loads(payload)

    # -- sync -----------------------------------------------------------------

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(
                    self._url, json=body, headers=self._headers, timeout=self.timeout
                )
            except httpx.HTTPError as e:
                if attempt < self.max_retries:
                    time.sleep(2.0 * 2**attempt)
                    continue
                raise MeragptAPIError(f"meraGPT request failed: {e}") from e
            if response.status_code == 200:
                return response.json()
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                time.sleep(_retry_wait(response, attempt))
                continue
            raise _error_from(response)
        raise AssertionError("unreachable")  # pragma: no cover

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._result(self._post(self._body(messages, stream=False)))

    def _stream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        body = self._body(messages, stream=True)
        for attempt in range(self.max_retries + 1):
            with httpx.stream(
                "POST", self._url, json=body, headers=self._headers, timeout=self.timeout
            ) as response:
                if response.status_code != 200:
                    response.read()
                    if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                        time.sleep(_retry_wait(response, attempt))
                        continue
                    raise _error_from(response)
                for line in response.iter_lines():
                    data = self._sse_data(line)
                    if data is None:
                        continue
                    chunk = self._chunk(data)
                    if chunk is None:
                        continue
                    if run_manager:
                        run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                    yield chunk
                return

    # -- async ----------------------------------------------------------------

    async def _apost(self, body: dict[str, Any]) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await http.post(self._url, json=body, headers=self._headers)
                except httpx.HTTPError as e:
                    if attempt < self.max_retries:
                        await asyncio.sleep(2.0 * 2**attempt)
                        continue
                    raise MeragptAPIError(f"meraGPT request failed: {e}") from e
                if response.status_code == 200:
                    return response.json()
                if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                    await asyncio.sleep(_retry_wait(response, attempt))
                    continue
                raise _error_from(response)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._result(await self._apost(self._body(messages, stream=False)))

    async def _astream(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: AsyncCallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        body = self._body(messages, stream=True)
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            for attempt in range(self.max_retries + 1):
                async with http.stream(
                    "POST", self._url, json=body, headers=self._headers
                ) as response:
                    if response.status_code != 200:
                        await response.aread()
                        if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                            await asyncio.sleep(_retry_wait(response, attempt))
                            continue
                        raise _error_from(response)
                    async for line in response.aiter_lines():
                        data = self._sse_data(line)
                        if data is None:
                            continue
                        chunk = self._chunk(data)
                        if chunk is None:
                            continue
                        if run_manager:
                            await run_manager.on_llm_new_token(chunk.text, chunk=chunk)
                        yield chunk
                    return


__all__ = ["ChatMeragpt", "DEFAULT_CHAT_MODEL"]
