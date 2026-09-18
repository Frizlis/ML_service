"""
FastAPI-приложение Inference-сервиса. Работает 24/7 (см. docker-compose:
restart: unless-stopped), подгружает модель из файла (артефакты версии —
в model_registry/, текущая версия и путь к ней — через src/registry).

Запуск: uv run uvicorn src.inference.main:app --host ... --port ... --workers 1
Один worker — намеренно: у каждого процесса uvicorn была бы своя копия
модели в памяти, и /model/switch обновил бы только один из них.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from src.common.logging_config import get_logger
from src.inference.model_service import ModelService
from src.inference.routes import router
from src.registry.registry import registry

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    model_service = ModelService(registry)
    # Best-effort: если модель ещё ни разу не обучали, приложение всё равно
    # должно подняться (см. ModelService.try_load_active) — /health покажет
    # model_loaded=false, /predict будет отвечать 503 до первого /model/switch.
    model_service.try_load_active()
    app.state.model_service = model_service
    yield


app = FastAPI(title="News Classifier Inference", lifespan=lifespan)
app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Последний рубеж обработки ошибок: непредвиденное исключение (баг, OOM
    и т.п.) не должно ронять процесс или отдавать клиенту сырой traceback —
    только чистый 500 и полная трассировка в лог сервера для разбора.
    """
    logger.exception("unhandled_exception", extra={"path": str(request.url)})
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})
