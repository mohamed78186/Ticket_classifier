"""The one class the rest of the app actually talks to.

TicketClassifier ties the three layers together: it turns a raw
ticket string into a prompt (prompts/), sends it through whatever LLM
implementation it was handed (llm/), and turns the raw reply back
into a validated, typed result (structured/). Everything upstream -
the CLI in app.py, the Streamlit UI, the evaluation harness - only
ever needs to know about this class.
"""

from prompts.classifier import build_ticket_classification_prompt
from structured.parser import parse_and_validate_ticket
from structured.schemas import TicketOutput

from .base import LLM
from .model import GenerationRequest, Message, MessageRole

_SYSTEM_INSTRUCTIONS = (
    "You classify support tickets. Return only valid JSON "
    "matching the requested schema."
)

# Deterministic output matters more than creativity here, and the
# response is short - a summary plus three labels - so this budget
# is generous rather than tight.
_MAX_RESPONSE_TOKENS = 200


class TicketClassifier:
    """Classifies a support ticket into category / sentiment / urgency."""

    def __init__(self, llm: LLM) -> None:
        self.llm = llm

    def classify(self, ticket: str) -> TicketOutput:
        self._require_non_empty_ticket(ticket)

        request = self._build_request(ticket)
        raw_reply = self.llm.generate(request)
        return parse_and_validate_ticket(raw_reply)

    @staticmethod
    def _require_non_empty_ticket(ticket: str) -> None:
        if not isinstance(ticket, str) or not ticket.strip():
            raise ValueError("ticket must be a non-empty string")

    @staticmethod
    def _build_request(ticket: str) -> GenerationRequest:
        prompt = build_ticket_classification_prompt(ticket)
        return GenerationRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=_SYSTEM_INSTRUCTIONS),
                Message(role=MessageRole.USER, content=prompt),
            ],
            temperature=0.0,
            max_new_tokens=_MAX_RESPONSE_TOKENS,
        )
