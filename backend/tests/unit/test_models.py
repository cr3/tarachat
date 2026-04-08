import pytest
from pydantic import ValidationError

from tarachat.models import (
    ChatMessage,
    ChatRequest,
    HealthResponse,
)


class TestChatMessage:
    def test_valid_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_assistant_role(self):
        msg = ChatMessage(role="assistant", content="Hi there")
        assert msg.role == "assistant"

    def test_missing_role_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(content="Hello")

    def test_missing_content_raises(self):
        with pytest.raises(ValidationError):
            ChatMessage(role="user")


class TestChatRequest:
    def test_valid_request(self):
        req = ChatRequest(message="Hello")
        assert req.message == "Hello"
        assert req.conversation_history == []

    def test_with_history(self):
        history = [{"role": "user", "content": "Hi"}]
        req = ChatRequest(message="Hello", conversation_history=history)
        assert len(req.conversation_history) == 1

    def test_empty_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="")

    def test_missing_message_raises(self):
        with pytest.raises(ValidationError):
            ChatRequest()


class TestHealthResponse:
    def test_healthy(self):
        resp = HealthResponse(status="healthy")
        assert resp.status == "healthy"
