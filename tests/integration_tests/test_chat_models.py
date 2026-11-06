"""Live tests for ChatMeragpt. Require `MERAGPT_API_KEY`."""

from __future__ import annotations

from langchain_meragpt import ChatMeragpt

DRAFT = (
    "The utilization of our platform facilitates the optimization of workflow "
    "efficiency across organizational boundaries."
)


def test_invoke() -> None:
    out = ChatMeragpt().invoke(DRAFT)
    assert isinstance(out.content, str) and out.content.strip()
    assert out.usage_metadata is not None
    assert "meragpt" in out.response_metadata


def test_stream() -> None:
    text = "".join(c.text for c in ChatMeragpt().stream(DRAFT))
    assert text.strip()


async def test_ainvoke() -> None:
    out = await ChatMeragpt().ainvoke(DRAFT)
    assert isinstance(out.content, str) and out.content.strip()
