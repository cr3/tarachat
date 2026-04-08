"""Integration tests for RAGSystem requiring real model downloads."""

from pathlib import Path

from tarachat.config import Settings
from tarachat.rag import RAGSystem


def test_create_initializes_all_components(tmp_path):
    """RAGSystem.create loads embeddings, vector store, tokenizer, and model."""
    settings = Settings(vector_store_path=str(tmp_path / "vs"))
    rag = RAGSystem.create(settings=settings, device="cpu")

    assert rag.embeddings is not None
    assert rag.vector_store is not None
    assert rag.tokenizer is not None
    assert rag.model is not None


def test_create_persists_vector_store(tmp_path):
    """RAGSystem.create saves the FAISS index to disk."""
    vs_path = tmp_path / "vs"
    settings = Settings(vector_store_path=str(vs_path))
    RAGSystem.create(settings=settings, device="cpu")

    assert (vs_path / "index.faiss").exists()


def test_create_loads_existing_vector_store(tmp_path):
    """RAGSystem.create reuses a previously saved vector store."""
    vs_path = tmp_path / "vs"
    settings = Settings(vector_store_path=str(vs_path))

    rag1 = RAGSystem.create(settings=settings, device="cpu")
    rag1.vector_store.save_local(str(vs_path))

    rag2 = RAGSystem.create(settings=settings, device="cpu")
    assert rag2.vector_store is not None
    assert Path(vs_path / "index.faiss").exists()
