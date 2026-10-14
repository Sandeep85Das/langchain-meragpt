"""Synchronous and asynchronous client for the meraGPT Decider API.

Decider 1 answers typed questions about a state over `POST /v1/systemone`
(https://meragpt.com/docs/decider). The request and response shapes are the
same as the TypeSafe / Jev decisions API.
"""

from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://meragpt.com"
DEFAULT_MODEL = "decider-1"
DEFAULT_TIMEOUT = 60.0
DEFAULT_MAX_RETRIES = 2
# Retry-After from a waking GPU can be 60s; never wait longer than this per try.
MAX_RETRY_WAIT = 60.0
RETRY_STATUSES = frozenset({429, 502, 503, 504})


class MeragptAPIError(Exception):
    """Raised when a meraGPT API request fails.

    Attributes:
        status: HTTP status code, or `None` for a transport error.
        code: The API's `error.code` (for example `insufficient_credit`).
        type: The API's `error.type` (for example `invalid_request_error`).
    """

    def __init__(
        self,
        message: str,
        *,
        status: int | None = None,
        code: str | None = None,
        type: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.type = type


def _error_from(response: httpx.Response) -> MeragptAPIError:
    try:
        err = response.json().get("error") or {}
    except ValueError:
        err = {}
    message = err.get("message") or response.text[:300] or response.reason_phrase
    return MeragptAPIError(
        f"meraGPT API error {response.status_code}: {message}",
        status=response.status_code,
        code=err.get("code"),
        type=err.get("type"),
    )


def _retry_wait(response: httpx.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    try:
        wait = float(header) if header is not None else 2.0 * 2**attempt
    except ValueError:
        wait = 2.0 * 2**attempt
    return min(max(wait, 0.0), MAX_RETRY_WAIT)


class MeragptClient:
    """Minimal client for the meraGPT API.

    Args:
        api_key: meraGPT API key (`mg-live-...`). Falls back to the
            `MERAGPT_API_KEY` environment variable.
        base_url: API base URL. Falls back to the `MERAGPT_BASE_URL`
            environment variable, then `https://meragpt.com`.
        model: Model id. Falls back to `MERAGPT_MODEL`, then `decider-1`.
        timeout: Request timeout in seconds.
        max_retries: Retries on 429 / 5xx capacity answers, honouring
            `Retry-After`. Nothing is charged for a request that is refused.
    """

    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        api_key = api_key or os.environ.get("MERAGPT_API_KEY")
        if not api_key:
            raise ValueError(
                "No meraGPT API key found. Pass `api_key=...` or set the "
                "MERAGPT_API_KEY environment variable. Create a key at "
                "https://meragpt.com/dashboard/keys"
            )
        self.api_key = api_key
        self.base_url = (base_url or os.environ.get("MERAGPT_BASE_URL") or DEFAULT_BASE_URL).rstrip(
            "/"
        )
        self.model = model or os.environ.get("MERAGPT_MODEL") or DEFAULT_MODEL
        self.timeout = timeout
        self.max_retries = max_retries

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "langchain-meragpt",
        }

    def _body(self, state: Any, questions: Mapping[str, Any], model: str | None) -> dict[str, Any]:
        if not questions:
            raise ValueError("At least one question is required.")
        return {
            "model": model or self.model,
            "state": state,
            "questions": dict(questions),
        }

    @staticmethod
    def _parse(response: httpx.Response) -> dict[str, Any]:
        try:
            data = response.json()
        except ValueError as e:
            raise MeragptAPIError(
                "meraGPT returned a non-JSON response", status=response.status_code
            ) from e
        if not isinstance(data, dict) or not isinstance(data.get("answers"), dict):
            raise MeragptAPIError(
                f"meraGPT returned an unexpected response: {data!r}",
                status=response.status_code,
            )
        return data

    def decide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Answer `questions` about `state`.

        Args:
            state: Any JSON value: a string, or an object such as
                `{"ticket": "...", "customer": {...}}`.
            questions: Maps a name to a question, for example
                `{"escalate": yes_no("Should a human take over?")}`.
            model: Overrides the client's model for this call.

        Returns:
            The full response: `answers[name]` holds each answer, and `usage`
            the billed tokens and cost.
        """
        body = self._body(state, questions, model)
        url = f"{self.base_url}/v1/systemone"
        for attempt in range(self.max_retries + 1):
            try:
                response = httpx.post(url, json=body, headers=self._headers, timeout=self.timeout)
            except httpx.HTTPError as e:
                if attempt < self.max_retries:
                    time.sleep(2.0 * 2**attempt)
                    continue
                raise MeragptAPIError(f"meraGPT request failed: {e}") from e
            if response.status_code == 200:
                return self._parse(response)
            if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                time.sleep(_retry_wait(response, attempt))
                continue
            raise _error_from(response)
        raise AssertionError("unreachable")  # pragma: no cover

    async def adecide(
        self,
        state: Any,
        questions: Mapping[str, Any],
        *,
        model: str | None = None,
    ) -> dict[str, Any]:
        """Async version of `decide`."""
        body = self._body(state, questions, model)
        url = f"{self.base_url}/v1/systemone"
        async with httpx.AsyncClient(timeout=self.timeout) as http:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await http.post(url, json=body, headers=self._headers)
                except httpx.HTTPError as e:
                    if attempt < self.max_retries:
                        await asyncio.sleep(2.0 * 2**attempt)
                        continue
                    raise MeragptAPIError(f"meraGPT request failed: {e}") from e
                if response.status_code == 200:
                    return self._parse(response)
                if response.status_code in RETRY_STATUSES and attempt < self.max_retries:
                    await asyncio.sleep(_retry_wait(response, attempt))
                    continue
                raise _error_from(response)
        raise AssertionError("unreachable")  # pragma: no cover


def _resolve_client(
    client: MeragptClient | None,
    api_key: str | None,
    base_url: str | None,
    model: str | None = None,
) -> MeragptClient:
    if client is not None:
        return client
    return MeragptClient(api_key=api_key, base_url=base_url, model=model)
