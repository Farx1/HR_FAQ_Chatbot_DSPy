"""
Golden-set regression tests: fixed inputs, deterministic expected outputs.
No model weights, no network beyond the embedding model download for retrieval.
"""

import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.server import is_hr_related  # noqa: E402

# question -> document that must appear in the top 3 retrieved chunks
RETRIEVAL_GOLDEN = {
    "How many vacation days do I get per year?": "leave_policy.md",
    "How much is the home office stipend for remote workers?": "remote_work_policy.md",
    "How does the 401k employer match vest?": "retirement_401k.md",
    "What is the nightly hotel limit on business trips?": "expense_reimbursement.md",
    "How do I report sexual harassment?": "harassment_policy.md",
    "When is the 90-day probation review?": "probation_period.md",
}

HR_GOLDEN = [
    "How many vacation days do I get?",
    "What is the dress code on Fridays?",
    "How do I submit an expense report?",
    "When is the annual bonus paid?",
    "Can I work remotely three days a week?",
    "What does the health insurance cover?",
]

OOD_GOLDEN = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "Will it rain tomorrow?",
    "How do I invest in bitcoin?",
    "Who won the football game?",
    "How do I fix my car engine?",
]

# Known misclassifications of the keyword filter, kept visible instead of hidden:
# the OOD questions of benchmark_professional.py that it accepts (measured).
KNOWN_OOD_MISSES = [
    "What is machine learning?",
    "What is cryptocurrency?",
    "What are the symptoms of flu?",
    "What is a healthy diet?",
]


@pytest.mark.parametrize("question", HR_GOLDEN)
def test_hr_question_accepted(question):
    assert is_hr_related(question)


@pytest.mark.parametrize("question", OOD_GOLDEN)
def test_ood_question_rejected(question):
    assert not is_hr_related(question)


@pytest.mark.parametrize("question", KNOWN_OOD_MISSES)
@pytest.mark.xfail(strict=True, reason="keyword OOD filter limitation")
def test_known_ood_misses(question):
    assert not is_hr_related(question)


@pytest.fixture(scope="module")
def rag(tmp_path_factory):
    pytest.importorskip("chromadb")
    pytest.importorskip("sentence_transformers")
    from backend.rag_engine import RAGEngine

    engine = RAGEngine(
        company_data_path="company_data/techcorp_solutions",
        persist_directory=str(tmp_path_factory.mktemp("db")),
    )
    assert engine.initialize(force_rebuild=True), "RAG init failed"
    return engine


@pytest.mark.parametrize("question,doc", RETRIEVAL_GOLDEN.items())
def test_retrieval_golden(rag, question, doc):
    top = [Path(h["source"]).name for h in rag.search(question, top_k=3)]
    assert doc in top, f"{doc} not in top 3: {top}"


class FakeRAG:
    """Fixed retrieval result, so /ask is tested without embeddings or a model."""

    is_initialized = True
    context = "[Leave Policy - 2. Paid Time Off]\n## 2. Paid Time Off\n- 20 days for 0-2 years of service\n- 25 days for 3-5 years"
    sources = [
        {
            "title": "Leave Policy - 2. Paid Time Off",
            "snippet": "20 days for 0-2 years",
            "category": "policy",
            "similarity": 0.81,
        }
    ]

    def __init__(self):
        self.calls = []

    def get_company_name(self):
        return "TechCorp Solutions"

    def get_context_for_question(self, question, mode="policy"):
        self.calls.append(question)
        return self.context, self.sources


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient

    import backend.server as server

    fake = FakeRAG()
    monkeypatch.setattr(server, "rag_engine", fake)
    monkeypatch.setattr(server, "hr_module", None)
    # No startup event: it would load the real RAG engine and model.
    return TestClient(server.app), fake


def test_ask_answers_from_retrieved_context(client):
    http, fake = client
    r = http.post("/ask", json={"question": "How many vacation days do I get?"})
    assert r.status_code == 200
    data = r.json()
    assert data["ood_reject"] is False
    assert "- 20 days for 0-2 years of service" in data["answer"]
    assert data["sources"][0]["title"] == "Leave Policy - 2. Paid Time Off"
    assert data["company"] == "TechCorp Solutions"
    assert fake.calls == ["How many vacation days do I get?"]


def test_ask_rejects_ood_before_retrieval(client):
    http, fake = client
    data = http.post("/ask", json={"question": "What is the capital of France?"}).json()
    assert data["ood_reject"] is True
    assert fake.calls == []


def test_ask_rejects_empty_question(client):
    http, _ = client
    assert http.post("/ask", json={"question": "   "}).status_code == 400
