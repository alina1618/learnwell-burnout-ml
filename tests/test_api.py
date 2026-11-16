from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_root():
    """Сервер запускається і відповідає."""
    r = client.get("/")
    assert r.status_code == 200


def test_docs():
    """Swagger-документація доступна."""
    assert client.get("/docs").status_code == 200


def test_predict_invalid_category():
    """Неіснуюча спеціальність 'Law' відхиляється з кодом 422."""
    r = client.post("/predict", json={"students": [{"Major_Category": "Law"}]})
    assert r.status_code == 422


def test_predict_empty_request():
    """Порожній запит відхиляється з кодом 422."""
    r = client.post("/predict", json={})
    assert r.status_code == 422