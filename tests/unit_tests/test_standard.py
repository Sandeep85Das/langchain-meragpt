"""LangChain standard unit tests for the tool."""

from __future__ import annotations

from typing import Any

from langchain_tests.unit_tests import ToolsUnitTests

from langchain_meragpt import MeragptDecisionTool


class TestMeragptDecisionToolUnit(ToolsUnitTests):
    @property
    def tool_constructor(self) -> type[MeragptDecisionTool]:
        return MeragptDecisionTool

    @property
    def tool_constructor_params(self) -> dict[str, Any]:
        return {"api_key": "mg-live-test"}

    @property
    def tool_invoke_params_example(self) -> dict[str, Any]:
        return {
            "state": "I was charged twice for order A-1042.",
            "question": "What is the message about?",
            "kind": "choice",
            "options": ["billing", "shipping", "account"],
        }
