"""Optional path to schema-constrained generation via Outlines.

This is an alternative to "generate free text, then validate" (which
is what the OpenAI-compatible / Ollama path uses): here the schema is
enforced token-by-token during generation itself, at the cost of only
working with a locally loaded model.
"""

from typing import Any, Type

from .outlines_adapter import create_structured_model


class StructuredGeneration:
    """Generates output that is already guaranteed to match a schema."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self._outlines_model = create_structured_model(model, tokenizer)

    def generate(self, prompt: str, schema: Type[Any]) -> Any:
        return self._outlines_model(prompt, schema)
