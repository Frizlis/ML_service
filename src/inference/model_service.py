"""
ModelService инкапсулирует загруженную модель + токенизатор + текущую версию
и умеет "на лету" переключаться на другую версию без рестарта процесса —
именно это дёргает эндпоинт /model/switch.

Конкурентность: predict()/reload() физически выполняются в worker-threads
threadpool FastAPI (torch — синхронный, не async-код), поэтому синхронизация
идёт через threading.RLock, а не asyncio.Lock (который заблокировал бы event
loop). Смена модели — atomic swap: новая модель грузится с диска ВНЕ лока
(это тяжёлая I/O операция), лок берётся только на быструю подмену ссылок —
поэтому параллельные /predict блокируются лишь на микросекунды подмены
указателей, а не на всё время загрузки весов с диска.
"""
from __future__ import annotations

import threading

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.logging_config import get_logger
from src.common.preprocessing import tokenize_texts, validate_and_build_input_text
from src.inference.exceptions import ModelNotLoadedError
from src.registry.registry import ModelRegistry

logger = get_logger(__name__)


class ModelService:
    def __init__(self, registry: ModelRegistry):
        self._registry = registry
        self._lock = threading.RLock()
        self._model = None
        self._tokenizer = None
        self._version: str | None = None
        self._device: torch.device | None = None

    @property
    def is_loaded(self) -> bool:
        with self._lock:
            return self._model is not None

    @property
    def version(self) -> str | None:
        with self._lock:
            return self._version

    def try_load_active(self) -> None:
        """
        Best-effort загрузка активной версии — вызывается в lifespan при
        старте приложения. Если в регистре ещё нет активной модели (сервис
        ни разу не обучали), просто логирует warning: старт приложения при
        этом НЕ должен падать, сервис поднимается в состоянии "модель не
        загружена" (см. /health, /predict).
        """
        try:
            self.reload()
        except Exception:
            logger.warning("no_active_model_on_startup", exc_info=True)

    def reload(self, version: str | None = None) -> str:
        """
        Загружает указанную версию (или текущую активную из регистра, если
        version не передан) и атомарно подменяет модель в памяти. Бросает
        NoActiveModelError/ModelNotFoundError (из src.registry.exceptions),
        если версии не существует — их ловит routes.py и превращает в 404.
        """
        version = version or self._registry.get_active().version
        self._registry.get_version(version)  # валидирует существование версии в индексе
        model_path = self._registry.version_dir(version)

        logger.info("loading_model_version", extra={"version": version, "path": str(model_path)})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        model.to(device)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(model_path)

        with self._lock:
            self._model = model
            self._tokenizer = tokenizer
            self._version = version
            self._device = device

        logger.info("model_version_loaded", extra={"version": version, "device": str(device)})
        return version

    def predict(self, headline: str, short_description: str) -> tuple[str, float]:
        with self._lock:
            if self._model is None:
                raise ModelNotLoadedError(
                    "Модель ещё не загружена — обучите и задеплойте версию перед /predict"
                )
            model, tokenizer, device = self._model, self._tokenizer, self._device

        # validate_and_build_input_text кидает EmptyTextError, если текст
        # пуст после нормализации — намеренно вне лока: это чистая CPU-логика
        # без обращения к модели, незачем держать за собой лок.
        input_text = validate_and_build_input_text(headline, short_description)
        encoding = tokenize_texts(tokenizer, [input_text])
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        probabilities = torch.softmax(logits, dim=-1)[0]
        predicted_id = int(torch.argmax(probabilities).item())
        confidence = float(probabilities[predicted_id].item())
        category = model.config.id2label[predicted_id]
        return category, confidence
