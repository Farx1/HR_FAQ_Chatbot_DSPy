"""
Test OOD (Out-of-Domain) detection functionality
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.server import is_hr_related


def test_hr_related_question():
    """Test that HR-related questions are correctly identified"""
    hr_questions = [
        "How many vacation days do I get?",
        "What is the remote work policy?",
        "How do I update my bank details for payroll?",
        "What benefits are available?",
        "How do I request sick leave?",
    ]
    
    for question in hr_questions:
        assert is_hr_related(question), f"Should identify as HR: {question}"


def test_ood_question():
    """Test that out-of-domain questions are correctly rejected"""
    ood_questions = [
        "What is the capital of France?",
        "How do I bake a chocolate cake?",
        "What is the weather today?",
        "Explain quantum mechanics",
        "How do I install Python?",
    ]
    
    for question in ood_questions:
        assert not is_hr_related(question), f"Should reject as OOD: {question}"


def test_edge_cases():
    """Test edge cases for OOD detection"""
    # Empty string
    assert not is_hr_related(""), "Empty string should not be HR-related"
    
    # Very short question
    assert not is_hr_related("Hi"), "Very short question should not be HR-related"
    
    # Question with HR keyword but clearly not HR
    assert not is_hr_related("What is the capital of Paris?"), "Should reject despite 'capital'"

