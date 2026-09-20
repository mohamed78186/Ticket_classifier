"""A small, provider-independent exception hierarchy.

Every concrete provider (OpenAI-compatible, local, whatever comes
next) is responsible for catching its own library's errors and
re-raising one of these instead. That way the rest of the codebase -
the gateway, the classifier, the CLI - only ever needs to know about
this one family of exceptions.
"""


class LLMError(Exception):
    """Root of every error this package can raise on purpose."""


class AuthenticationError(LLMError):
    """Credentials were missing, malformed, or rejected by the provider."""


class InvalidRequestError(LLMError):
    """The provider considered the request itself invalid."""


class RateLimitError(LLMError):
    """Too many requests were sent in too little time."""


class ConnectionError(LLMError):  # noqa: A001 - intentional shadow, scoped to this module
    """The request never reached the provider, or timed out in transit."""


class ProviderError(LLMError):
    """The provider accepted the request but failed on its own end."""
