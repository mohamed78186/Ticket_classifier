"""Cross-cutting logging for anything that calls out to a model."""

import logging
import time
from functools import wraps
from typing import Any, Callable, TypeVar

_logger = logging.getLogger(__name__)
_Result = TypeVar("_Result")


def log_llm_call(function: Callable[..., _Result]) -> Callable[..., _Result]:
    """Wrap ``function`` so every call is timed and logged.

    Success and failure are both logged with the elapsed time in
    milliseconds; failures additionally include the exception type
    and a full traceback via ``logger.exception``. The original
    exception is always re-raised unchanged.
    """

    @wraps(function)
    def timed_call(*args: Any, **kwargs: Any) -> _Result:
        start = time.perf_counter()
        _logger.info("llm_call_started function=%s", function.__qualname__)

        try:
            outcome = function(*args, **kwargs)
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            _logger.exception(
                "llm_call_failed function=%s error_type=%s elapsed_ms=%.2f",
                function.__qualname__,
                type(exc).__name__,
                elapsed_ms,
            )
            raise

        elapsed_ms = (time.perf_counter() - start) * 1000
        _logger.info(
            "llm_call_succeeded function=%s elapsed_ms=%.2f",
            function.__qualname__,
            elapsed_ms,
        )
        return outcome

    return timed_call
