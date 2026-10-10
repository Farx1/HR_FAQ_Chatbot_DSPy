"""
Test RAG (Retrieval-Augmented Generation) functionality
"""

import sys
import os

import pytest

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_rag_engine_import():
    """Test that RAG engine can be imported"""
    try:
        from backend.rag_engine import get_rag_engine, RAGEngine

        assert callable(get_rag_engine) and callable(RAGEngine)
    except ImportError as e:
        # If dependencies are missing, skip test
        import pytest

        pytest.skip(f"RAG dependencies not available: {e}")


def test_rag_engine_initialization():
    """Test that RAG engine can be created (initialize() falls back without deps)"""
    from backend.rag_engine import get_rag_engine

    rag_engine = get_rag_engine(company_data_path="company_data/techcorp_solutions")
    assert rag_engine is not None, "RAG engine should be created"


def test_rag_context_retrieval():
    """Test that RAG can retrieve context for questions"""
    # Only a missing optional dependency is a valid reason to skip.
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    from backend.rag_engine import get_rag_engine

    rag_engine = get_rag_engine(company_data_path="company_data/techcorp_solutions")
    assert rag_engine.is_initialized, "RAG dependencies installed but init failed"

    question = "How many vacation days do I get?"
    context, sources = rag_engine.get_context_for_question(question, mode="policy")

    assert isinstance(context, str), "Context should be a string"
    assert isinstance(sources, list), "Sources should be a list"
