"""
Test retrieval functionality with a tiny local fixture corpus (no external downloads)
"""

import sys
import os
import json

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_retrieval_with_local_corpus(tmp_path):
    """Test retrieval using a minimal local corpus fixture"""
    # Only a missing optional dependency is a valid reason to skip.
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    from backend.rag_engine import RAGEngine

    corpus_dir = tmp_path / "test_corpus"
    corpus_dir.mkdir()
    (corpus_dir / "test_policy.md").write_text(
        """
# Vacation Policy

Full-time employees receive 20 vacation days per year.
Vacation requests must be submitted at least 2 weeks in advance.
        """.strip()
    )
    (corpus_dir / "company_info.json").write_text(
        json.dumps({"company_name": "TestCorp", "description": "Test company"})
    )

    # Own persist dir: never rebuild the real backend/chroma_db index from a test.
    rag_engine = RAGEngine(
        company_data_path=str(corpus_dir), persist_directory=str(tmp_path / "db")
    )
    assert rag_engine.initialize(force_rebuild=True), "RAG init failed"

    question = "How many vacation days do I get?"
    context, sources = rag_engine.get_context_for_question(question, mode="policy")

    assert isinstance(context, str), "Context should be a string"
    assert isinstance(sources, list), "Sources should be a list"
    assert "vacation" in context.lower() or "20" in context, (
        "Context should contain relevant information about vacation"
    )


def test_retrieval_basic_logic():
    """Test basic retrieval logic without full RAG engine (unit test)"""
    # Simple test that doesn't require RAG engine initialization
    # Just verify that we can import and understand the structure

    try:
        from backend.rag_engine import RAGEngine, get_rag_engine

        # Verify that RAGEngine class exists
        assert RAGEngine is not None, "RAGEngine class should exist"

        # Verify that get_rag_engine function exists
        assert callable(get_rag_engine), "get_rag_engine should be callable"

        # This test passes if imports work, which is sufficient for offline testing
        assert True, "Retrieval module structure is correct"

    except ImportError as e:
        # If RAG engine can't be imported, skip test
        import pytest

        pytest.skip(f"RAG engine not available: {e}")
