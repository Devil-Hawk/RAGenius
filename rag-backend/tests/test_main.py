from fastapi.testclient import TestClient
from main import app
# import pytest  # F401 - Keep commented or remove if truly unused

client = TestClient(app)


def test_upload_no_files():
    response = client.post("/upload", files=[])
    assert response.status_code == 422


def test_ask_no_documents():
    response = client.post("/ask", data={"question": "test question"})
    assert response.status_code == 400
    assert response.json()["error"] == "No documents uploaded."


def test_rate_limiting():
    # Make multiple requests quickly
    for _ in range(3):
        response = client.post("/ask", data={"question": "test question"})
    assert response.status_code == 429
    assert "Too many requests" in response.json()["detail"] 