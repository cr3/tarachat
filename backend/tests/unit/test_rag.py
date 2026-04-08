"""Unit tests for RAGSystem pure methods (no ML models required)."""

import json
from unittest.mock import MagicMock

import pytest
from langchain.docstore.document import Document

from tarachat.config import Settings
from tarachat.rag import RAGSystem


@pytest.fixture
def rag(tmp_path):
    """RAGSystem with explicit settings and no heavy initialization."""
    settings = Settings(vector_store_path=str(tmp_path / "vs"))
    return RAGSystem(settings=settings, device="cpu")


class TestIsReady:
    def test_not_ready_at_construction(self, rag):
        assert rag.is_ready() is False

    def test_ready_when_all_set(self, rag):
        rag.embeddings = object()
        rag.vector_store = object()
        rag.model = object()
        rag.tokenizer = object()
        assert rag.is_ready() is True

    def test_not_ready_missing_model(self, rag):
        rag.embeddings = object()
        rag.vector_store = object()
        rag.tokenizer = object()
        assert rag.is_ready() is False

    def test_not_ready_missing_embeddings(self, rag):
        rag.vector_store = object()
        rag.model = object()
        rag.tokenizer = object()
        assert rag.is_ready() is False


class TestBuildPrompt:
    def test_basic_prompt(self, rag):
        docs = [Document(page_content="Some context")]
        result = rag._build_prompt("What is X?", docs)
        assert "Some context" in result
        assert "Question : What is X?" in result
        assert "Réponse :" in result

    def test_prompt_without_history(self, rag):
        docs = [Document(page_content="ctx")]
        result = rag._build_prompt("Q?", docs)
        assert "Historique" not in result

    def test_prompt_with_history(self, rag):
        docs = [Document(page_content="ctx")]
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
        result = rag._build_prompt("Q?", docs, history)
        assert "Historique de la conversation" in result
        assert "Utilisateur: Hi" in result
        assert "Assistant: Hello" in result

    def test_prompt_truncates_history_to_6(self, rag):
        docs = [Document(page_content="ctx")]
        history = [{"role": "user", "content": f"msg{i}"} for i in range(10)]
        result = rag._build_prompt("Q?", docs, history)
        # Only last 6 should appear
        assert "msg4" in result
        assert "msg9" in result
        assert "msg3" not in result

    def test_multiple_docs_joined(self, rag):
        docs = [
            Document(page_content="First doc"),
            Document(page_content="Second doc"),
        ]
        result = rag._build_prompt("Q?", docs)
        assert "First doc" in result
        assert "Second doc" in result


class TestBuildDemoResponse:
    def test_with_docs(self, rag):
        docs = [Document(page_content="Important content here")]
        result = rag._build_demo_response(docs)
        assert "Important content here" in result
        assert "trouvé" in result

    def test_with_two_docs(self, rag):
        docs = [
            Document(page_content="First"),
            Document(page_content="Second"),
        ]
        result = rag._build_demo_response(docs)
        assert "First" in result
        assert "Second" in result

    def test_no_docs(self, rag):
        result = rag._build_demo_response([])
        assert "Désolé" in result

    def test_long_doc_truncated(self, rag):
        docs = [Document(page_content="x" * 500)]
        result = rag._build_demo_response(docs)
        assert len(result) < 500


class TestExtractSources:
    def test_source_preview(self, rag):
        docs = [Document(page_content="Hello world from document")]
        sources = rag._extract_sources(docs)
        assert len(sources) == 1
        assert sources[0].endswith("...")

    def test_multiple_sources(self, rag):
        docs = [
            Document(page_content="Doc A"),
            Document(page_content="Doc B"),
        ]
        sources = rag._extract_sources(docs)
        assert len(sources) == 2

    def test_long_source_truncated(self, rag):
        docs = [Document(page_content="x" * 200)]
        sources = rag._extract_sources(docs)
        assert len(sources[0]) == 103  # 100 chars + "..."


class TestRetrieveDocuments:
    def test_none_vector_store_returns_empty(self, rag):
        assert rag.vector_store is None
        assert rag.retrieve_documents("query") == []

    def test_empty_index_returns_empty(self, rag):
        mock_vs = MagicMock()
        mock_vs.index.ntotal = 0
        rag.vector_store = mock_vs
        assert rag.retrieve_documents("query") == []

    def test_uses_settings_top_k(self, rag):
        mock_vs = MagicMock()
        mock_vs.index.ntotal = 5
        mock_vs.similarity_search.return_value = [Document(page_content="hit")]
        rag.vector_store = mock_vs
        result = rag.retrieve_documents("query")
        mock_vs.similarity_search.assert_called_once_with("query", k=rag.settings.top_k)
        assert len(result) == 1

    def test_custom_k(self, rag):
        mock_vs = MagicMock()
        mock_vs.index.ntotal = 5
        mock_vs.similarity_search.return_value = []
        rag.vector_store = mock_vs
        rag.retrieve_documents("query", k=7)
        mock_vs.similarity_search.assert_called_once_with("query", k=7)


class TestAddDocuments:
    def test_empty_texts_is_noop(self, rag):
        rag.add_documents([])
        # No error, no vector store interaction

    def test_adds_and_saves(self, rag, tmp_path):
        rag.settings.vector_store_path = str(tmp_path / "vs")
        mock_vs = MagicMock()
        rag.vector_store = mock_vs
        rag.settings.chunk_size = 5000  # large enough to avoid splitting
        rag.add_documents(["hello world"], [{"source": "test"}])
        mock_vs.add_documents.assert_called_once()
        docs = mock_vs.add_documents.call_args[0][0]
        assert any("hello world" in d.page_content for d in docs)
        mock_vs.save_local.assert_called_once()

    def test_splits_long_text(self, rag, tmp_path):
        rag.settings.vector_store_path = str(tmp_path / "vs")
        mock_vs = MagicMock()
        rag.vector_store = mock_vs
        rag.settings.chunk_size = 50
        rag.settings.chunk_overlap = 0
        long_text = "word " * 100  # 500 chars
        rag.add_documents([long_text])
        docs = mock_vs.add_documents.call_args[0][0]
        assert len(docs) > 1  # Should be chunked


class TestChat:
    def test_demo_mode(self, rag):
        mock_vs = MagicMock()
        mock_vs.index.ntotal = 1
        mock_vs.similarity_search.return_value = [
            Document(page_content="Found this info"),
        ]
        rag.vector_store = mock_vs
        rag.settings.demo_mode = True
        response, sources = rag.chat("question")
        assert "Found this info" in response
        assert len(sources) == 1

    def test_demo_mode_no_docs(self, rag):
        rag.settings.demo_mode = True
        response, sources = rag.chat("question")
        assert "Désolé" in response
        assert sources == []


class TestChatStream:
    def test_demo_mode_yields_sse(self, rag):
        mock_vs = MagicMock()
        mock_vs.index.ntotal = 1
        mock_vs.similarity_search.return_value = [
            Document(page_content="Doc content"),
        ]
        rag.vector_store = mock_vs
        rag.settings.demo_mode = True
        events = list(rag.chat_stream("hello"))
        assert len(events) == 3
        token_event = json.loads(events[0].removeprefix("data: ").strip())
        assert token_event["type"] == "token"
        sources_event = json.loads(events[1].removeprefix("data: ").strip())
        assert sources_event["type"] == "sources"
        assert events[2].strip() == "data: [DONE]"

    def test_demo_mode_no_docs(self, rag):
        rag.settings.demo_mode = True
        events = list(rag.chat_stream("hello"))
        token_event = json.loads(events[0].removeprefix("data: ").strip())
        assert "Désolé" in token_event["content"]
