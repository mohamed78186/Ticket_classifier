"""A retrying decorator-as-a-class around any LLM implementation.

LLMGateway itself implements the LLM interface, so it can be dropped
in anywhere a plain provider is expected - the only difference is
that transient failures (rate limits, dropped connections, server
errors) get a few extra chances before giving up.
"""

import logging
import random
import time
from collections.abc import Callable

from .base import LLM
from .decorators import log_llm_call
from .exceptions import ConnectionError, LLMError, ProviderError, RateLimitError
from .model import GenerationRequest

_logger = logging.getLogger(__name__)

# Failures worth retrying. Anything else (bad auth, malformed request, ...)
# is not going to fix itself just because we waited a bit.
_RETRYABLE_ERRORS = (RateLimitError, ConnectionError, ProviderError)


class LLMGateway(LLM):
    """Wraps another LLM and retries it with exponential backoff + jitter."""

    def __init__(
        self,
        llm: LLM,
        max_attempts: int = 3,
        base_delay: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
        random_uniform: Callable[[float, float], float] = random.uniform,
    ) -> None:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if base_delay < 0:
            raise ValueError("base_delay must be >= 0")

        self.llm = llm
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self._sleep = sleep
        self._random_uniform = random_uniform

    @log_llm_call
    def generate(self, request: GenerationRequest) -> str:
        last_attempt = self.max_attempts

        for attempt_number in range(1, last_attempt + 1):
            try:
                return self.llm.generate(request)
            except LLMError as exc:
                if not isinstance(exc, _RETRYABLE_ERRORS):
                    raise
                if attempt_number == last_attempt:
                    _logger.error("llm_retry_exhausted attempts=%s", attempt_number)
                    raise
                self._back_off(attempt_number, exc)

        # Defensive only - the loop above always returns or raises.
        raise RuntimeError("unreachable retry state")

    def _back_off(self, attempt_number: int, exc: Exception) -> None:
        ceiling = self.base_delay * (2 ** (attempt_number - 1))
        delay = self._random_uniform(0, ceiling)
        _logger.warning(
            "llm_retrying attempt=%s next_attempt=%s delay_seconds=%.3f error=%s",
            attempt_number,
            attempt_number + 1,
            delay,
            type(exc).__name__,
        )
        self._sleep(delay)
