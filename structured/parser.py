"""Turn a model's raw text reply into a trustworthy TicketOutput."""

import json

from pydantic import ValidationError

from .schemas import TicketOutput


def parse_and_validate_ticket(raw_output: str) -> TicketOutput:
    """Parse ``raw_output`` as JSON and validate it against TicketOutput.

    Raises ``ValueError`` (never a bare JSON or pydantic exception) so
    callers only ever need to catch one error type regardless of
    which step failed.
    """
    _require_non_blank_string(raw_output)

    parsed = _load_json(raw_output)
    return _validate(parsed)


def _require_non_blank_string(raw_output: str) -> None:
    if not isinstance(raw_output, str) or not raw_output.strip():
        raise ValueError("model output must be a non-empty string")


def _load_json(raw_output: str):
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as exc:
        raise ValueError("model output is not valid JSON") from exc


def _validate(data) -> TicketOutput:
    try:
        return TicketOutput.model_validate(data)
    except ValidationError as exc:
        raise ValueError("model output does not match TicketOutput") from exc
