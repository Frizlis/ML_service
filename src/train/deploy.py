"""
Деплой новой версии модели: активация в регистре + best-effort уведомление
Inference-сервиса о необходимости подхватить новую версию.
"""
from __future__ import annotations

import httpx

from src.common.config import settings
from src.common.logging_config import get_logger
from src.registry.registry import ModelRegistry

logger = get_logger(__name__)


def deploy_version(registry: ModelRegistry, version_id: str) -> None:
    """
    Активация версии в регистре ДОЛЖНА пройти всегда — registry.json это
    источник истины. HTTP-вызов к Inference — лишь "быстрый путь" для
    горячего обновления без рестарта контейнера: если Inference сейчас
    недоступен (например, ещё не поднят), это НЕ должно ронять весь
    train-pipeline — цель обучения (валидная версия в регистре) уже
    достигнута, а Inference при своём следующем старте сам прочитает
    активную версию из registry.json.
    """
    registry.activate(version_id)

    url = f"{settings.train_inference_url}/model/switch"
    try:
        response = httpx.post(
            url, json={"version": version_id}, timeout=settings.train_deploy_timeout_seconds
        )
        response.raise_for_status()
        logger.info("inference_notified", extra={"version": version_id, "url": url})
    except httpx.HTTPError as exc:
        logger.warning(
            "inference_notification_failed",
            extra={"version": version_id, "url": url, "error": str(exc)},
        )
