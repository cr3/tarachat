"""Unit tests for config module."""

from tarachat.config import Settings, get_settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.host == "0.0.0.0"
        assert s.port == 8000
        assert s.demo_mode is True
        assert s.chunk_size == 512
        assert s.chunk_overlap == 50
        assert s.top_k == 3
        assert s.max_tokens == 128

    def test_cors_origins_default(self):
        s = Settings()
        assert s.cors_origins == "http://localhost:5173"


class TestGetSettings:
    def test_cached(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
