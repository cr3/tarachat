import json


class TestRootEndpoint:
    def test_root_returns_welcome(self, client):
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Welcome to TaraChat API"
        assert "docs" in data
        assert "health" in data


class TestHealthEndpoint:
    def test_healthy_status(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


def _parse_sse_events(response_text: str) -> list:
    """Parse SSE events from response text."""
    events = []
    for line in response_text.strip().split('\n'):
        line = line.strip()
        if line.startswith('data: ') and line != 'data: [DONE]':
            events.append(json.loads(line[6:]))
    return events


class TestChatEndpoint:
    def test_successful_chat_stream(self, client):
        response = client.post("/chat", json={"message": "Hello"})
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        events = _parse_sse_events(response.text)
        token_events = [e for e in events if e["type"] == "token"]
        source_events = [e for e in events if e["type"] == "sources"]
        assert len(token_events) >= 1
        assert token_events[0]["content"] == "Echo: Hello"
        assert len(source_events) == 1
        assert source_events[0]["sources"] == ["doc1"]

    def test_chat_with_history(self, client):
        payload = {
            "message": "Follow up",
            "conversation_history": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi!"}
            ]
        }
        response = client.post("/chat", json=payload)
        assert response.status_code == 200

    def test_chat_empty_message_returns_422(self, client):
        response = client.post("/chat", json={"message": ""})
        assert response.status_code == 422

    def test_chat_missing_message_returns_422(self, client):
        response = client.post("/chat", json={})
        assert response.status_code == 422
