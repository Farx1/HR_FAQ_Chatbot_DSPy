"""POST /rebuild-index must refuse callers without the admin token."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient  # noqa: E402

from backend.server import app  # noqa: E402

client = TestClient(app)


def test_rebuild_index_disabled_without_admin_token(monkeypatch):
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert client.post("/rebuild-index").status_code == 403


def test_rebuild_index_rejects_wrong_token(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert client.post("/rebuild-index").status_code == 401
    assert (
        client.post("/rebuild-index", headers={"X-Admin-Token": "nope"}).status_code
        == 401
    )
