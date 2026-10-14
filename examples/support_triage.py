"""Triage support tickets with Decider 1. Needs MERAGPT_API_KEY.

python examples/support_triage.py
"""

from langchain_meragpt import MeragptDecider, choice, score, yes_no

TICKETS = [
    "I was charged twice for order A-1042. Fix it today or I'm cancelling.",
    "My parcel says delivered but it isn't here.",
    "How do I change the email on my account?",
]

decider = MeragptDecider(
    questions={
        "team": choice(
            "Which team should handle this ticket?",
            {
                "billing": "Charges, refunds or invoices",
                "shipping": "Delivery of an order",
                "account": "Login, settings or profile changes",
            },
        ),
        "churn_risk": yes_no("Is the customer threatening to leave?"),
        "priority": score("How urgent is this ticket?", ["low", "medium", "high"]),
    }
)

for ticket in TICKETS:
    a = decider.invoke({"state": {"ticket": ticket}})
    print(
        f"{a['team']['choice']:9} churn={a['churn_risk']['noul']:.2f} "
        f"priority={a['priority']['score']:.1f}  {ticket}"
    )
