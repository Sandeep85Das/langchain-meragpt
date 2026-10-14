# langchain-meragpt

LangChain integration for **meraGPT Decider 1**, a decision model. Ask a typed question
about any text or JSON state (yes/no, pick one, or a score on a scale) and get back
calibrated probabilities, not free text.

Decider 1 is state of the art among zero-shot models on the
[typed-decisions benchmark](https://huggingface.co/datasets/LocalLLaMA/typed-decisions)
on Hugging Face. It costs **$0.04 per 1M input tokens**, and output is free.

This package gives you:

| | what it does |
|---|---|
| `MeragptDecider` | Decider 1 as a LangChain `Runnable` |
| `MeragptDecisionTool` | a tool an agent can call to classify, route or approve |
| `MeragptDecisionMiddleware` | answers your questions about the conversation once per run and keeps the answers in agent state |
| `MeragptRouterMiddleware` | picks the model for each call |
| `MeragptToolApprovalMiddleware` | approves or blocks a tool call before it runs |
| `MeragptStopMiddleware` | stops an agent that is looping or has given up |

## Installation

```bash
pip install langchain-meragpt
export MERAGPT_API_KEY=mg-live-...
```

Create a key at [meragpt.com/dashboard/keys](https://meragpt.com/dashboard/keys). New
accounts get $1 of free credit.

## Quickstart

```python
from langchain_meragpt import MeragptDecider, choice, score, yes_no

decider = MeragptDecider()
answers = decider.invoke({
    "state": {"ticket": "I was charged twice for order A-1042. Fix it today or I'm cancelling."},
    "questions": {
        "topic": choice("What is the ticket mainly about?", {
            "billing": "Charges, refunds or invoices",
            "shipping": "Delivery of an order",
            "login": "Signing in or account access",
        }),
        "churn_risk": yes_no("Is the customer threatening to leave?"),
        "urgency": score("How urgent is this ticket?", ["low", "medium", "high"]),
    },
})

answers["topic"]["choice"]          # "billing"
answers["topic"]["probabilities"]   # {"billing": 0.96, "shipping": 0.035, "login": 0.004}
answers["churn_risk"]["noul"]       # probability of yes (0.5 here: a threat, but a conditional one)
answers["urgency"]["score"]         # expected level, 0 = low ... 2 = high (1.39)
```

Question types:

- `yes_no(question)`: the answer is `{"type": "noul", "noul": p_yes}`.
- `choice(question, options)`: up to 10 options, as a list of labels or a mapping of
  label to a one-line description (descriptions help). The answer has `choice`,
  `confidence` and `probabilities`.
- `score(question, levels)`: an ordered scale, lowest first, up to 10 levels. The
  answer has `score` (the expected level), `legend` and `probabilities`.

Up to 64 questions can go in one call, and they are all answered against the same state.
`state` can be a string or any JSON object.

## Middleware

All middleware fails open: if a meraGPT call fails, the agent carries on as though the
middleware weren't there, and a warning is logged. `MeragptToolApprovalMiddleware` can
fail closed instead.

### Triage and escalation (`MeragptDecisionMiddleware`)

```python
from langchain.agents import create_agent
from langchain_meragpt import MeragptDecisionMiddleware, choice, yes_no

triage = MeragptDecisionMiddleware(questions={
    "topic": choice("What is the customer asking about?", {
        "billing": "Charges, refunds or invoices",
        "technical": "Bugs, errors or outages",
        "account": "Login, settings or cancellation",
    }),
    "escalate": yes_no("Is the customer asking for a human or threatening to leave?"),
})

agent = create_agent("openai:gpt-6-luna", tools=[...], middleware=[triage])
result = agent.invoke({"messages": [{"role": "user", "content": "..."}]})

if result["meragpt_decisions"]["escalate"]["noul"] > 0.7:
    hand_off_to_a_human(result)
```

Pass `inject_into_prompt=True` to add a one-line summary of the answers to the system
message, so the agent can act on them too.

### Approve or block tool calls (`MeragptToolApprovalMiddleware`)

```python
from langchain_meragpt import MeragptToolApprovalMiddleware

guard = MeragptToolApprovalMiddleware(tools=["run_shell", "send_email"], threshold=0.5)
agent = create_agent("openai:gpt-6-luna", tools=[run_shell, send_email, search],
                     middleware=[guard])
```

Before a targeted tool runs, Decider sees the user's latest request, the tool's name and
description, and the arguments, and answers *"Does this tool call do what the user asked
for?"*. Below the threshold, the tool doesn't run and the model gets an error
`ToolMessage` explaining why. You can pass your own `question`, where "yes" approves the
call. Keep it narrow, because one specific question separates far better than a
composite "is this unsafe or unwanted?".

### Route between models (`MeragptRouterMiddleware`)

```python
from langchain_meragpt import MeragptRouterMiddleware

router = MeragptRouterMiddleware(routes={
    "simple": ("A short factual question or small talk", "openai:gpt-6-luna"),
    "complex": ("Multi-step reasoning, code or analysis", "anthropic:claude-opus-5-5"),
}, default="complex", min_confidence=0.6)

agent = create_agent("openai:gpt-6-luna", tools=[...], middleware=[router])
```

### Stop a stuck agent (`MeragptStopMiddleware`)

```python
from langchain_meragpt import MeragptStopMiddleware

agent = create_agent("openai:gpt-6-luna", tools=[...],
                     middleware=[MeragptStopMiddleware(min_tool_calls=4, threshold=0.7)])
```

After each model call that asks for more tool calls, the recent trace is checked:
*"Is the agent stuck repeating the same failing step?"* and *"Has the agent given up on
the task?"*. If either question reaches the threshold, the run ends with a short
"Stopped early" message instead of spending more calls. Pass your own `questions` to
check something else.

## As a tool

```python
from langchain_meragpt import MeragptDecisionTool

agent = create_agent("openai:gpt-6-luna", tools=[MeragptDecisionTool()])
```

The agent calls `meragpt_decide` with `state`, `question`, `kind` (`yes_no`, `choice` or
`score`) and `options`, and gets JSON back, for example `{"p_yes": 0.83}`.

## Async

Every class has an async path (`ainvoke`, `adecide`, and the async middleware hooks).

## What gets sent

Each decision sends the state and your questions to `POST https://meragpt.com/v1/systemone`.
The middleware sends only what it needs:

- **Decision and stop middleware:** the last 20 turns.
- **Approval middleware:** the user's latest message, the tool call and the tool's
  description.
- **Router middleware:** the user's latest message.

Long texts are clipped. See the
[privacy policy](https://meragpt.com/privacy) for how requests are handled.

## Configuration

| environment variable | default | |
|---|---|---|
| `MERAGPT_API_KEY` | (required) | your API key, `mg-live-...` |
| `MERAGPT_BASE_URL` | `https://meragpt.com` | API base URL, for a proxy or gateway |
| `MERAGPT_MODEL` | `decider-1` | model id |

Every class also takes `api_key=` and `base_url=`, or a shared `client=MeragptClient(...)`.
Requests refused with 429 or 503 while capacity scales up are retried, honouring
`Retry-After`. Nothing is charged for a refused request.

## Development

```bash
uv sync --all-groups
make test               # unit tests, network disabled
make integration_tests  # live API, needs MERAGPT_API_KEY
make lint
```

## Links

- API reference: [meragpt.com/docs/decider](https://meragpt.com/docs/decider)
- Model card, limits and evaluation: [meragpt.com/models/decider-1](https://meragpt.com/models/decider-1)
- Launch post: [Introducing Decider 1](https://meragpt.com/blog/introducing-decider-1)
