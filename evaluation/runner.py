"""A lighter-weight evaluation path: any classify() callable, category accuracy only.

Unlike TicketEvaluator (evaluator.py) this doesn't need a generator +
schema, just a plain function from ticket text to a result with a
``.category`` attribute - handy for quick experiments or unit tests.
"""

from collections.abc import Callable, Iterable

from .models import ClassificationMetrics


def evaluate_classifier(
    rows: Iterable[dict], classify: Callable[[str], object]
) -> ClassificationMetrics:
    """Score ``classify`` against the ``category`` field of each row."""
    total = 0
    true_labels: list[str] = []
    predicted_labels: list[str] = []

    for row in rows:
        total += 1
        try:
            prediction = classify(row["message"])
        except (KeyError, ValueError, TypeError, AttributeError):
            continue
        true_labels.append(row["category"])
        predicted_labels.append(prediction.category)

    valid = len(true_labels)
    correct = sum(t == p for t, p in zip(true_labels, predicted_labels))

    return ClassificationMetrics(
        total=total,
        valid_outputs=valid,
        accuracy=(correct / valid) if valid else None,
        macro_f1=_macro_f1(true_labels, predicted_labels) if valid else None,
    )


def _macro_f1(true_labels: list[str], predicted_labels: list[str]) -> float:
    """Unweighted mean of the per-label F1 score."""
    labels = sorted(set(true_labels) | set(predicted_labels))
    if not labels:
        return 0.0

    per_label_scores = [_f1_for_label(label, true_labels, predicted_labels) for label in labels]
    return sum(per_label_scores) / len(per_label_scores)


def _f1_for_label(label: str, true_labels: list[str], predicted_labels: list[str]) -> float:
    true_positives = sum(t == label and p == label for t, p in zip(true_labels, predicted_labels))
    false_positives = sum(t != label and p == label for t, p in zip(true_labels, predicted_labels))
    false_negatives = sum(t == label and p != label for t, p in zip(true_labels, predicted_labels))

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) else 0.0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)
