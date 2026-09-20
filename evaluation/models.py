"""Data containers shared by the evaluation harness.

Note: ``EvaluationResult`` was previously imported by evaluator.py and
metrics.py but was never actually defined anywhere in this module,
which meant importing either of those files raised an ImportError.
It's defined here now, with the fields both call sites expect.
"""

from dataclasses import dataclass

from structured.schemas import TicketOutput


@dataclass
class EvaluationResult:
    """Outcome of evaluating a single ticket against its ground truth."""

    ticket_id: object
    success: bool

    expected_category: str
    expected_sentiment: str
    expected_urgency: str

    result: TicketOutput | None = None

    category_correct: bool = False
    sentiment_correct: bool = False
    urgency_correct: bool = False

    error: str | None = None


@dataclass(frozen=True)
class ClassificationMetrics:
    """Aggregate accuracy figures produced by evaluation/runner.py."""

    total: int
    valid_outputs: int
    accuracy: float | None = None
    macro_f1: float | None = None
