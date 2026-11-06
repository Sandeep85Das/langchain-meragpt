"""LangChain standard unit tests for the chat model."""

from __future__ import annotations

from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_meragpt import ChatMeragpt


class TestChatMeragptUnit(ChatModelUnitTests):
    @property
    def chat_model_class(self) -> type[ChatMeragpt]:
        return ChatMeragpt

    @property
    def chat_model_params(self) -> dict:
        return {"model": "restyler-1", "api_key": "mg-live-test"}

    @property
    def init_from_env_params(self) -> tuple[dict, dict, dict]:
        return (
            {"MERAGPT_API_KEY": "mg-live-env", "MERAGPT_BASE_URL": "https://proxy.example"},
            {},
            {"api_key": "mg-live-env", "base_url": "https://proxy.example"},
        )

    @property
    def has_tool_calling(self) -> bool:
        return False

    @property
    def has_structured_output(self) -> bool:
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        return True
