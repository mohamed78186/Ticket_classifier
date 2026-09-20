"""Aggregate statistics computed over a list of EvaluationResult objects."""

from collections.abc import Sequence

from .models import EvaluationResult


def _succeeded(results: Sequence[EvaluationResult]) -> list[EvaluationResult]:
    """Keep only the cases where structured generation actually produced a result."""
    return [item for item in results if item.success]


def validity_rate(results: Sequence[EvaluationResult]) -> float:
    """Share of cases where structured generation succeeded at all."""
    if not results:
        return 0.0
    return sum(item.success for item in results) / len(results)


def category_accuracy(results: Sequence[EvaluationResult]) -> float:
    """Category accuracy, computed only over successfully generated cases."""
    successes = _succeeded(results)
    if not successes:
        return 0.0
    return sum(item.category_correct for item in successes) / len(successes)


def sentiment_accuracy(results: Sequence[EvaluationResult]) -> float:
    """Sentiment accuracy, computed only over successfully generated cases."""
    successes = _succeeded(results)
    if not successes:
        return 0.0
    return sum(item.sentiment_correct for item in successes) / len(successes)


def urgency_accuracy(results: Sequence[EvaluationResult]) -> float:
    """Urgency accuracy, computed only over successfully generated cases."""
    successes = _succeeded(results)
    if not successes:
        return 0.0
    return sum(item.urgency_correct for item in successes) / len(successes)


def overall_success_rate(results: Sequence[EvaluationResult]) -> float:
    """Share of cases that generated successfully AND got all three labels right."""
    if not results:
        return 0.0

    fully_correct = 0
    for item in results:
        if item.success and item.category_correct and item.sentiment_correct and item.urgency_correct:
            fully_correct += 1

    return fully_correct / len(results)


def failure_count(results: Sequence[EvaluationResult]) -> int:
    """How many cases failed to produce a usable structured result."""
    return sum(not item.success for item in results)
