"""
FastAPI dependency, отдающая ModelService из app.state — используется в
роутах через Depends(get_model_service). Хранение в app.state (а не в
module-level синглтоне) даёт изоляцию между тестами: TestClient создаёт
своё приложение с чистым app.state, и app.dependency_overrides позволяет
подменить сервис на мок без единого реального обращения к модели.
"""
from fastapi import Request

from src.inference.model_service import ModelService


def get_model_service(request: Request) -> ModelService:
    return request.app.state.model_service
