"""The one shape a ticket classification is allowed to have.

Anything a model produces has to pass through this schema before the
rest of the app will touch it. ``extra="forbid"`` means a model that
adds a stray field (or hallucinates a new one) fails validation
loudly instead of quietly polluting downstream data.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["technical", "account", "delivery", "billing", "subscription"]
Sentiment = Literal["positive", "negative", "neutral"]
Urgency = Literal["low", "medium", "high"]


class TicketOutput(BaseModel):
    """Validated result of classifying one support ticket."""

    model_config = ConfigDict(extra="forbid")

    category: Category
    sentiment: Sentiment
    urgency: Urgency
    summary: str = Field(min_length=1, max_length=1000)
