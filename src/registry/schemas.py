"""
Pydantic-схема содержимого model_registry/registry.json.

Зачем схема, а не просто dict: файл читает и пишет только Train (создаёт
версии, активирует), но читает ещё и Inference (при старте контейнера) —
если формат файла случайно разъедется (например, кто-то вручную поправит
JSON), pydantic сразу же кинет понятную ошибку валидации при чтении, а не
уронит сервис в middle of request с невнятным KeyError.
"""
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class ModelVersionEntry(BaseModel):
    version: str
    created_at: datetime
    metrics: dict[str, float]
    status: Literal["active", "archived"] = "archived"


class RegistryIndex(BaseModel):
    active_version: str | None = None
    versions: dict[str, ModelVersionEntry] = Field(default_factory=dict)
