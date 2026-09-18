"""
HTTP-эндпоинты Inference-сервиса: health check, предсказание, смена модели.
"""
from fastapi import APIRouter, Depends, HTTPException, status

from src.common.exceptions import EmptyTextError
from src.common.logging_config import get_logger
from src.inference.dependencies import get_model_service
from src.inference.exceptions import ModelNotLoadedError
from src.inference.model_service import ModelService
from src.inference.schemas import (
    HealthResponse,
    PredictRequest,
    PredictResponse,
    SwitchModelRequest,
    SwitchModelResponse,
)
from src.registry.exceptions import ModelNotFoundError, NoActiveModelError

logger = get_logger(__name__)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(model_service: ModelService = Depends(get_model_service)) -> HealthResponse:
    """
    Всегда 200 — сигнализирует, что процесс жив (liveness). model_loaded
    отдельно сигнализирует готовность отвечать на /predict (readiness) —
    это осознанное разделение: "процесс работает" и "модель готова" не одно
    и то же, особенно до самого первого обучения.
    """
    return HealthResponse(
        status="ok",
        model_loaded=model_service.is_loaded,
        model_version=model_service.version,
    )


@router.post("/predict", response_model=PredictResponse)
def predict(
    body: PredictRequest, model_service: ModelService = Depends(get_model_service)
) -> PredictResponse:
    try:
        category, confidence = model_service.predict(body.headline, body.short_description)
    except ModelNotLoadedError as exc:
        # 503, а не 500: отсутствие модели — ожидаемое состояние системы до
        # первого обучения/деплоя, а не баг.
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)) from exc
    except EmptyTextError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail=str(exc)
        ) from exc

    return PredictResponse(
        category=category, confidence=confidence, model_version=model_service.version
    )


@router.post("/model/switch", response_model=SwitchModelResponse)
def switch_model(
    body: SwitchModelRequest, model_service: ModelService = Depends(get_model_service)
) -> SwitchModelResponse:
    """
    Переключает загруженную модель на указанную версию (или на текущую
    активную из регистра, если version не передан). Работает и когда модель
    ещё ни разу не была загружена — именно этим эндпоинтом Train бутстрапит
    первую версию сразу после успешного обучения (см. src/train/deploy.py).
    """
    previous_version = model_service.version
    try:
        new_version = model_service.reload(body.version)
    except (ModelNotFoundError, NoActiveModelError) as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    logger.info(
        "model_switched",
        extra={"previous_version": previous_version, "new_version": new_version},
    )
    return SwitchModelResponse(model_version=new_version, previous_version=previous_version)
