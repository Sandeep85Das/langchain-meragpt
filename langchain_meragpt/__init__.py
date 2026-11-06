"""LangChain integration for meraGPT: Decider 1 decisions and the Restyler 1 chat model."""

from importlib import metadata

from langchain_meragpt._client import MeragptAPIError, MeragptClient
from langchain_meragpt.chat_models import ChatMeragpt
from langchain_meragpt.decider import MeragptDecider
from langchain_meragpt.middleware import (
    MeragptDecisionMiddleware,
    MeragptDecisionState,
    MeragptRouterMiddleware,
    MeragptStopMiddleware,
    MeragptToolApprovalMiddleware,
    conversation_state,
)
from langchain_meragpt.questions import choice, probability, score, yes_no
from langchain_meragpt.tools import MeragptDecisionTool

try:
    __version__ = metadata.version(__package__)
except metadata.PackageNotFoundError:
    __version__ = ""
del metadata

__all__ = [
    "ChatMeragpt",
    "MeragptAPIError",
    "MeragptClient",
    "MeragptDecider",
    "MeragptDecisionMiddleware",
    "MeragptDecisionState",
    "MeragptDecisionTool",
    "MeragptRouterMiddleware",
    "MeragptStopMiddleware",
    "MeragptToolApprovalMiddleware",
    "choice",
    "conversation_state",
    "probability",
    "score",
    "yes_no",
    "__version__",
]
