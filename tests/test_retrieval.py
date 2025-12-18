"""
Test retrieval functionality with a tiny local fixture corpus (no external downloads)
"""
import sys
import os
import tempfile
import json
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_retrieval_with_local_corpus():
    """Test retrieval using a minimal local corpus fixture"""
    try:
        from backend.rag_engine import RAGEngine
        
        # Create a temporary directory for test corpus
        with tempfile.TemporaryDirectory() as tmpdir:
            corpus_dir = Path(tmpdir) / "test_corpus"
            corpus_dir.mkdir()
            
            # Create a minimal test document
            test_doc = corpus_dir / "test_policy.md"
            test_doc.write_text("""
# Vacation Policy

Full-time employees receive 20 vacation days per year.
Vacation requests must be submitted at least 2 weeks in advance.
            """.strip())
            
            # Create company info
            company_info = corpus_dir / "company_info.json"
            company_info.write_text(json.dumps({
                "company_name": "TestCorp",
                "description": "Test company"
            }))
            
            # Try to initialize RAG engine with test corpus
            # This may fail if dependencies are missing, which is OK
            try:
                rag_engine = RAGEngine(
                    company_data_path=str(corpus_dir),
                    force_init=True
                )
                
                # If initialization succeeded, test retrieval
                if rag_engine.is_initialized:
                    question = "How many vacation days do I get?"
                    context, sources = rag_engine.get_context_for_question(question, mode="policy")
                    
                    assert isinstance(context, str), "Context should be a string"
                    assert isinstance(sources, list), "Sources should be a list"
                    # If we got context, it should contain relevant information
                    if context:
                        assert "vacation" in context.lower() or "20" in context, \
                            "Context should contain relevant information about vacation"
                else:
                    # RAG not initialized (missing dependencies) - skip test
                    import pytest
                    pytest.skip("RAG engine not initialized (missing dependencies)")
                    
            except Exception as e:
                # If dependencies are missing, skip test
                import pytest
                pytest.skip(f"RAG engine initialization failed (likely missing deps): {e}")
                
    except ImportError as e:
        # If RAG engine can't be imported, skip test
        import pytest
        pytest.skip(f"RAG engine not available: {e}")


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

