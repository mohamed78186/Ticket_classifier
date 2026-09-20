"""The contract every model backend must satisfy.

Anything that can turn a GenerationRequest into text - a hosted API,
a locally-loaded model, a gateway wrapping another backend - should
inherit from LLM. Nothing else in the project should care which one
it's actually talking to.
"""

from abc import ABC, abstractmethod

from .model import GenerationRequest


class LLM(ABC):
    """Minimal interface for anything that can generate text."""

    @abstractmethod
    def generate(self, request: GenerationRequest) -> str:
        """Return the raw text completion for ``request``.

        Implementations are expected to raise one of the exceptions
        defined in ``llm.exceptions`` on failure, rather than letting
        backend-specific errors leak out.
        """
        raise NotImplementedError
