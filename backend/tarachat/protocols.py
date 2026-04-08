"""Protocols for dependency injection and testing."""

from __future__ import annotations

from collections.abc import Generator
from typing import Any, Protocol, runtime_checkable

from langchain.docstore.document import Document


@runtime_checkable
class RAGProtocol(Protocol):
    """Public interface for the RAG system.

    Implemented by RAGSystem (production) and FakeRAGSystem (tests).
    """

    model: Any
    vector_store: Any

    def initialize(self) -> None: ...

    def is_ready(self) -> bool: ...

    def add_documents(
        self, texts: list[str], metadatas: list[dict] | None = None,
    ) -> None: ...

    def retrieve_documents(
        self, query: str, k: int | None = None,
    ) -> list[Document]: ...

    def chat_stream(
        self, message: str, conversation_history: list[dict] | None = None,
    ) -> Generator[str, None, None]: ...

    def create_empty_vector_store(self) -> Any: ...
