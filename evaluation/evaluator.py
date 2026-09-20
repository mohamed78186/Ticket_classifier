"""Runs the classifier over a labeled dataset and records what happened.

TicketEvaluator only produces EvaluationResult rows - it deliberately
does not compute accuracy or any other aggregate; that's the job of
evaluation/metrics.py. Splitting the two makes it possible to re-run
the metrics on a saved set of results without calling the model again.
"""

import pandas as pd

from prompts.classifier import build_ticket_classifier
from structured.parser import parse_and_validate_ticket
from structured.schemas import TicketOutput

from .models import EvaluationResult


class TicketEvaluator:
    """Evaluates a ticket-classification generator against ground truth."""

    def __init__(self, generator):
        self.generator = generator

    def evaluate_ticket(self, row) -> EvaluationResult:
        """Run the full pipeline for one row and record the outcome.

        Each stage (prompt building, generation, parsing) is isolated
        in its own try/except so a failure anywhere still produces a
        usable EvaluationResult with a specific error message, instead
        of an unhandled exception aborting the whole dataset.
        """
        ticket_id = row["ticket_id"]

        try:
            prompt = build_ticket_classifier(row["message"])
        except Exception as exc:
            return self._failed(row, f"Prompt generation failed: {type(exc).__name__}: {exc}")

        try:
            raw_output = self.generator.generate(prompt, TicketOutput)
            print(f"[{ticket_id}] RAW RESULT: {raw_output}")
        except Exception as exc:
            return self._failed(row, f"Generation failed: {type(exc).__name__}: {exc}")

        try:
            prediction = parse_and_validate_ticket(raw_output)
            print(f"[{ticket_id}] PARSED RESULT: {prediction}")
        except Exception as exc:
            return self._failed(row, f"Parsing/validation failed: {type(exc).__name__}: {exc}")

        return EvaluationResult(
            ticket_id=ticket_id,
            success=True,
            expected_category=row["category"],
            expected_sentiment=row["sentiment"],
            expected_urgency=row["urgency"],
            result=prediction,
            category_correct=prediction.category == row["category"],
            sentiment_correct=prediction.sentiment == row["sentiment"],
            urgency_correct=prediction.urgency == row["urgency"],
            error=None,
        )

    @staticmethod
    def _failed(row, message: str) -> EvaluationResult:
        return EvaluationResult(
            ticket_id=row["ticket_id"],
            success=False,
            expected_category=row["category"],
            expected_sentiment=row["sentiment"],
            expected_urgency=row["urgency"],
            result=None,
            error=message,
        )

    def evaluate(self, dataframe) -> list[EvaluationResult]:
        """Evaluate every row of ``dataframe`` and return the raw results."""
        return [self.evaluate_ticket(row) for _, row in dataframe.iterrows()]

    def evaluate_dataframe(self, dataframe) -> pd.DataFrame:
        """Evaluate ``dataframe`` and flatten the results into a report table."""
        results = self.evaluate(dataframe)
        return pd.DataFrame(self._as_report_row(item) for item in results)

    @staticmethod
    def _as_report_row(item: EvaluationResult) -> dict:
        prediction = item.result if (item.success and item.result is not None) else None

        return {
            "ticket_id": item.ticket_id,
            "expected_category": item.expected_category,
            "expected_sentiment": item.expected_sentiment,
            "expected_urgency": item.expected_urgency,
            "predicted_category": prediction.category if prediction else None,
            "predicted_sentiment": prediction.sentiment if prediction else None,
            "predicted_urgency": prediction.urgency if prediction else None,
            "category_correct": item.category_correct if item.success else None,
            "sentiment_correct": item.sentiment_correct if item.success else None,
            "urgency_correct": item.urgency_correct if item.success else None,
            "success": item.success,
            "error": item.error,
        }
