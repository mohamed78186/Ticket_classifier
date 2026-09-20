"""Backward-compatible alias for the (correctly spelled) openai_compatible module.

Kept around only so old imports using the misspelled module name keep
working; new code should import from ``openai_compatible`` directly.
"""

from .openai_compatible import OpenAICompatible

__all__ = ["OpenAICompatible"]
