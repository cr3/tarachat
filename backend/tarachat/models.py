
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message", min_length=1)
    conversation_history: list[ChatMessage] | None = Field(
        default_factory=list,
        description="Previous conversation history"
    )


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
