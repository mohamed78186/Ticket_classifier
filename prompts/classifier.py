"""Prompt for turning a raw support ticket into structured JSON."""

from .base import build_prompt

_ALLOWED_CATEGORIES = ("technical", "account", "delivery", "billing", "subscription")
_ALLOWED_SENTIMENTS = ("positive", "negative", "neutral")
_ALLOWED_URGENCIES = ("low", "medium", "high")


def build_ticket_classification_prompt(ticket: str) -> str:
    """Build the prompt asking the model to classify ``ticket``."""
    return build_prompt(
        role="You are a precise customer-support ticket classification assistant.",
        task=(
            "Classify the ticket and produce JSON with exactly these fields: "
            "category, sentiment, urgency, summary."
        ),
        user_input=ticket,
        constraints=[
            f"category must be one of: {', '.join(_ALLOWED_CATEGORIES)}",
            f"sentiment must be one of: {', '.join(_ALLOWED_SENTIMENTS)}",
            f"urgency must be one of: {', '.join(_ALLOWED_URGENCIES)}",
            "summary must be concise and factual",
            "Return JSON only; do not use markdown or explanations",
        ],
    )


# Older parts of the project (and evaluation/evaluator.py) still refer
# to this prompt builder by its original, shorter name.
build_ticket_classifier = build_ticket_classification_prompt
