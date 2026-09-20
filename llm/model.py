"""Plain data containers describing a single call to a language model.

These types intentionally know nothing about *how* a model is reached
(HTTP, local weights, whatever) - that's the job of the classes in
base.py and providers/. Keeping the request shape provider-agnostic
means we can swap the backend without touching this module at all.
"""

from dataclasses import dataclass, field
from enum import Enum


class MessageRole(str, Enum):
    """Who is "speaking" in a chat-style prompt."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Message:
    """One turn in the conversation sent to the model."""

    role: MessageRole
    content: str


@dataclass(frozen=True)
class GenerationRequest:
    """Everything a provider needs in order to produce a completion.

    The object is frozen on purpose: once a request has been built it
    should not be mutated in-flight (for example while it sits inside
    a retry loop in the gateway).
    """

    messages: list[Message] = field(default_factory=list)
    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None
    max_new_tokens: int = 100

    def __post_init__(self) -> None:
        problem = self._first_validation_error()
        if problem is not None:
            raise ValueError(problem)

    def _first_validation_error(self) -> str | None:
        if len(self.messages) == 0:
            return "messages must not be empty"
        if self.temperature < 0:
            return "temperature must be >= 0"
        if self.top_p is not None and not (0 < self.top_p <= 1):
            return "top_p must be in (0, 1]"
        if self.top_k is not None and self.top_k < 1:
            return "top_k must be >= 1"
        if self.max_new_tokens < 1:
            return "max_new_tokens must be >= 1"
        return None
