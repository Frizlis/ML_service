"""
Юнит-тесты HTTP-контракта Inference: /health, /predict, /model/switch.
ModelService подменяется рукописным фейком через app.dependency_overrides —
ни один тест не грузит реальную модель и не трогает диск.
"""
import pytest
from fastapi.testclient import TestClient

from src.common.exceptions import EmptyTextError
from src.inference.dependencies import get_model_service
from src.inference.exceptions import ModelNotLoadedError
from src.inference.main import app
from src.registry.exceptions import ModelNotFoundError, NoActiveModelError


class _FakeModelService:
    """Дублёр ModelService с ровно тем интерфейсом, что используют роуты."""

    def __init__(self):
        self._loaded = False
        self._version = None
        self.predict_result = ("POLITICS", 0.9)
        self.predict_exception = None
        self.reload_exception = None
        self.reload_calls = []

    @property
    def is_loaded(self):
        return self._loaded

    @property
    def version(self):
        return self._version

    def predict(self, headline, short_description):
        if self.predict_exception:
            raise self.predict_exception
        return self.predict_result

    def reload(self, version=None):
        self.reload_calls.append(version)
        if self.reload_exception:
            raise self.reload_exception
        self._loaded = True
        self._version = version or "active-version"
        return self._version


@pytest.fixture
def fake_service():
    return _FakeModelService()


@pytest.fixture
def client(fake_service):
    app.dependency_overrides[get_model_service] = lambda: fake_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


# --- /health ---


def test_health_when_model_not_loaded(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_loaded": False, "model_version": None}


def test_health_when_model_loaded(fake_service, client):
    fake_service._loaded = True
    fake_service._version = "v1"
    response = client.get("/health")
    assert response.json() == {"status": "ok", "model_loaded": True, "model_version": "v1"}


# --- /predict ---


def test_predict_returns_503_when_model_not_loaded(fake_service, client):
    fake_service.predict_exception = ModelNotLoadedError("no model")
    response = client.post("/predict", json={"headline": "H", "short_description": "D"})
    assert response.status_code == 503


def test_predict_returns_prediction_when_model_loaded(fake_service, client):
    fake_service._loaded = True
    fake_service._version = "v1"
    fake_service.predict_result = ("SPORTS", 0.75)
    response = client.post("/predict", json={"headline": "H", "short_description": "D"})
    assert response.status_code == 200
    assert response.json() == {"category": "SPORTS", "confidence": 0.75, "model_version": "v1"}


def test_predict_returns_422_when_text_empty_after_normalization(fake_service, client):
    fake_service.predict_exception = EmptyTextError("empty")
    response = client.post("/predict", json={"headline": "H", "short_description": "D"})
    assert response.status_code == 422


def test_predict_returns_422_on_missing_field(client):
    response = client.post("/predict", json={"headline": "H"})
    assert response.status_code == 422


def test_predict_returns_422_on_empty_string_field(client):
    response = client.post("/predict", json={"headline": "", "short_description": "D"})
    assert response.status_code == 422


# --- /model/switch ---


def test_switch_model_success(fake_service, client):
    response = client.post("/model/switch", json={"version": "v2"})
    assert response.status_code == 200
    assert response.json() == {"model_version": "v2", "previous_version": None}
    assert fake_service.reload_calls == ["v2"]


def test_switch_model_without_version_uses_active(fake_service, client):
    response = client.post("/model/switch", json={})
    assert response.status_code == 200
    assert fake_service.reload_calls == [None]


def test_switch_model_returns_404_on_unknown_version(fake_service, client):
    fake_service.reload_exception = ModelNotFoundError("not found")
    response = client.post("/model/switch", json={"version": "does-not-exist"})
    assert response.status_code == 404


def test_switch_model_returns_404_when_registry_empty(fake_service, client):
    fake_service.reload_exception = NoActiveModelError("no active model")
    response = client.post("/model/switch", json={})
    assert response.status_code == 404
