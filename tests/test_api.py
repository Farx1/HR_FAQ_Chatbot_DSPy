"""
API tests using FastAPI TestClient (no running server required)
"""
import sys
import os
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_health_endpoint():
    """Test health endpoint using FastAPI TestClient"""
    from fastapi.testclient import TestClient
    from backend.server import app
    
    client = TestClient(app)
    response = client.get("/health")
    
    assert response.status_code == 200, f"Health endpoint should return 200, got {response.status_code}"
    data = response.json()
    assert "status" in data, "Health response should contain 'status'"
    assert data["status"] == "healthy", "Status should be 'healthy'"


def test_ask_endpoint_structure():
    """Test that ask endpoint has correct structure (without actually calling it)"""
    from backend.server import AskRequest, AskResponse, app
    
    # Verify request model fields (Pydantic v2 uses model_fields)
    assert "question" in AskRequest.model_fields, "AskRequest should have 'question' field"
    assert "mode" in AskRequest.model_fields, "AskRequest should have 'mode' field"
    
    # Verify response model fields
    assert "answer" in AskResponse.model_fields, "AskResponse should have 'answer' field"
    assert "ood_reject" in AskResponse.model_fields, "AskResponse should have 'ood_reject' field"
    
    # Verify app has the endpoint
    routes = [route.path for route in app.routes]
    assert "/ask" in routes, "App should have /ask endpoint"
    assert "/health" in routes, "App should have /health endpoint"


def test_ask_endpoint_with_mock():
    """Test /ask endpoint with mocked model calls (offline test)"""
    from fastapi.testclient import TestClient
    import backend.server
    
    # Create a mock RAG engine instance
    mock_rag_inst = MagicMock()
    mock_rag_inst.is_initialized = False  # This will skip RAG context retrieval
    mock_rag_inst.get_company_name.return_value = "TechCorp Solutions"
    
    # Mock the global variables
    with patch.object(backend.server, 'hr_module', None), \
         patch.object(backend.server, 'rag_engine', mock_rag_inst):
        
        client = TestClient(backend.server.app)
        
        # Test HR-related question (should not be rejected)
        response = client.post(
            "/ask",
            json={"question": "How many vacation days do I get?", "mode": "policy"}
        )
        
        assert response.status_code == 200, f"Ask endpoint should return 200, got {response.status_code}"
        data = response.json()
        
        # Verify response structure
        assert "answer" in data, "Response should contain 'answer'"
        assert "ood_reject" in data, "Response should contain 'ood_reject'"
        assert isinstance(data["answer"], str), "Answer should be a string"
        assert isinstance(data["ood_reject"], bool), "ood_reject should be a boolean"
        
        # HR question should not be rejected
        assert data["ood_reject"] is False, "HR question should not be rejected"
        assert len(data["answer"]) > 0, "Answer should not be empty"


def test_ask_endpoint_ood_detection():
    """Test /ask endpoint correctly rejects out-of-domain questions"""
    from fastapi.testclient import TestClient
    import backend.server
    
    # Create a mock RAG engine instance
    mock_rag_inst = MagicMock()
    mock_rag_inst.is_initialized = False  # This will skip RAG context retrieval
    mock_rag_inst.get_company_name.return_value = "TechCorp Solutions"
    
    # Mock the global variables
    with patch.object(backend.server, 'hr_module', None), \
         patch.object(backend.server, 'rag_engine', mock_rag_inst):
        
        client = TestClient(backend.server.app)
        
        # Test OOD question (should be rejected)
        response = client.post(
            "/ask",
            json={"question": "What is the capital of France?", "mode": "policy"}
        )
        
        assert response.status_code == 200, f"Ask endpoint should return 200, got {response.status_code}"
        data = response.json()
        
        # OOD question should be rejected
        assert data["ood_reject"] is True, "OOD question should be rejected"
        assert "answer" in data, "Response should contain 'answer' even when rejected"


def test_ask_endpoint_demo_mode():
    """Test POST /ask endpoint returns JSON with 'answer' when DEMO_MODE=1 (hr_module is None)"""
    from fastapi.testclient import TestClient
    import backend.server
    
    # Create a mock RAG engine instance
    mock_rag_inst = MagicMock()
    mock_rag_inst.is_initialized = False  # This will skip RAG context retrieval
    mock_rag_inst.get_company_name.return_value = "TechCorp Solutions"
    
    # Mock the global variables to simulate DEMO_MODE (hr_module is None)
    with patch.object(backend.server, 'hr_module', None), \
         patch.object(backend.server, 'rag_engine', mock_rag_inst):
        
        client = TestClient(backend.server.app)
        
        # Test HR-related question (should return answer via fallback)
        response = client.post(
            "/ask",
            json={"question": "How many vacation days do I get?", "mode": "policy"}
        )
        
        assert response.status_code == 200, f"Ask endpoint should return 200, got {response.status_code}"
        data = response.json()
        
        # Verify response structure - must have 'answer' field
        assert "answer" in data, "Response must contain 'answer' field"
        assert isinstance(data["answer"], str), "Answer must be a string"
        assert len(data["answer"]) > 0, "Answer must not be empty"
        
        # Verify other expected fields
        assert "ood_reject" in data, "Response should contain 'ood_reject'"
        assert isinstance(data["ood_reject"], bool), "ood_reject must be a boolean"
