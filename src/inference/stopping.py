"""Detects whether generation has just produced one of several stop sequences."""


class StopSequenceDetector:
    """Checks a running list of generated token ids against known stop sequences."""

    def __init__(self, stop_token_sequences: list[list[int]] | None = None):
        self._sequences = stop_token_sequences or []

    def match(self, generated_tokens: list[int]) -> int:
        """Return the length of the stop sequence found at the end of
        ``generated_tokens``, or 0 if none of them match yet."""
        for sequence in self._sequences:
            if not sequence:
                continue
            tail = generated_tokens[-len(sequence):]
            if len(generated_tokens) >= len(sequence) and tail == sequence:
                return len(sequence)
        return 0
