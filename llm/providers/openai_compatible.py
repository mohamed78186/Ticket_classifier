"""LLM backend for anything speaking the OpenAI chat-completions dialect.

This is deliberately generic: OpenAI itself, Groq, and Ollama's
``/v1`` compatibility endpoint all understand the same request shape,
so a single adapter covers every one of them - only ``base_url`` and
the API key change between deployments.
"""

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError as _SdkAuthenticationError,
    BadRequestError as _SdkBadRequestError,
    InternalServerError as _SdkInternalServerError,
    OpenAI,
    RateLimitError as _SdkRateLimitError,
)

from ..base import LLM
from ..config import LLMSettings
from ..exceptions import (
    AuthenticationError,
    ConnectionError,
    InvalidRequestError,
    ProviderError,
    RateLimitError,
)
from ..model import GenerationRequest


class OpenAICompatible(LLM):
    """Talks to any OpenAI-compatible ``/chat/completions`` endpoint."""

    def __init__(self, settings: LLMSettings) -> None:
        self._client = OpenAI(
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout=settings.timeout,
        )
        self._model_name = settings.model

    def generate(self, request: GenerationRequest) -> str:
        payload = self._to_payload(request)
        try:
            response = self._client.chat.completions.create(**payload)
        except _SdkAuthenticationError as exc:
            raise AuthenticationError("provider authentication failed") from exc
        except _SdkBadRequestError as exc:
            raise InvalidRequestError("provider rejected the request") from exc
        except _SdkRateLimitError as exc:
            raise RateLimitError("provider rate limit reached") from exc
        except (APIConnectionError, APITimeoutError) as exc:
            raise ConnectionError("provider connection failed") from exc
        except _SdkInternalServerError as exc:
            raise ProviderError("provider server error") from exc

        content = response.choices[0].message.content
        if content is None:
            raise ProviderError("provider returned empty content")
        return content

    def _to_payload(self, request: GenerationRequest) -> dict:
        if request.top_k is not None:
            raise InvalidRequestError(
                "top_k is not supported by the OpenAI-compatible chat adapter"
            )

        payload: dict = {
            "model": self._model_name,
            "messages": [
                {"role": turn.role.value, "content": turn.content}
                for turn in request.messages
            ],
            "temperature": request.temperature,
            "max_tokens": request.max_new_tokens,
        }
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        return payload

    # Kept for anything that still reaches into ``.model`` /
    # ``.client`` directly (e.g. notebooks written against the
    # earlier version of this class).
    @property
    def model(self) -> str:
        return self._model_name

    @property
    def client(self) -> OpenAI:
        return self._client
