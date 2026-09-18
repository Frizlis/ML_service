"""
Pydantic-модели запросов/ответов Inference API. FastAPI сам валидирует
входящие данные по этим схемам и возвращает 422 при несовпадении типов/
границ — отдельный ручной парсинг тела запроса не нужен.
"""
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """
    Вход зеркалит структуру обучающих данных (headline + short_description):
    модель обучалась на input_text = headline + " [SEP] " + short_description
    (см. src/common/preprocessing.py), и предсказание обязано готовиться той
    же функцией — иначе распределение входа разъедется с тем, что модель
    видела на обучении.
    """

    headline: str = Field(..., min_length=1, max_length=1000)
    short_description: str = Field(..., min_length=1, max_length=5000)


class PredictResponse(BaseModel):
    category: str
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    model_version: str | None = None


class SwitchModelRequest(BaseModel):
    # version не передан -> подхватить текущую активную версию из регистра.
    version: str | None = None


class SwitchModelResponse(BaseModel):
    model_version: str
    previous_version: str | None = None
