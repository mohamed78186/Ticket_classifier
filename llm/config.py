"""Runtime configuration for whichever OpenAI-compatible endpoint we talk to.

Values are read from environment variables (or a local ``.env`` file)
so the exact same code can point at Groq, OpenAI, a self-hosted
gateway, or Ollama's ``/v1`` endpoint - only the environment changes,
never this class.
"""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEFAULT_BASE_URL = "https://api.groq.com/openai/v1"


class LLMSettings(BaseSettings):
    """Connection details pulled from ``LLM_*`` environment variables."""

    api_key: str = Field(..., min_length=1, validation_alias="LLM_API_KEY")
    base_url: str = Field(_DEFAULT_BASE_URL, min_length=1, validation_alias="LLM_BASE_URL")
    model: str = Field(..., min_length=1, validation_alias="LLM_MODEL")

    # Retry behaviour for the gateway - see llm/gateway.py.
    max_attempts: int = Field(3, ge=1, validation_alias="LLM_MAX_ATTEMPTS")
    base_delay: float = Field(1.0, ge=0, validation_alias="LLM_BASE_DELAY")

    # Per-request network timeout, in seconds.
    timeout: float = Field(30.0, gt=0, validation_alias="LLM_TIMEOUT")

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )
