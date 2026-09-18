# Inference-сервис: работает 24/7 (см. docker-compose.yml: restart: unless-stopped).
# GPU здесь не резервируется — torch с CUDA-библиотеками спокойно работает и
# на CPU, просто не использует GPU, если он не проброшен в контейнер. Это
# осознанное упрощение: держать 24/7-сервис с постоянно занятой GPU-картой
# ради редких единичных запросов на классификацию — расточительно.
FROM python:3.11-slim

# uv ставим как обычный pip-пакет — самый простой путь для Docker-слоя, не
# нужно тянуть отдельный installer-скрипт или curl | sh внутрь образа.
RUN pip install --no-cache-dir uv

WORKDIR /app

# Сначала только файлы зависимостей — Docker закэширует слой с uv sync и не
# будет часами переустанавливать torch/transformers при каждом изменении
# кода сервиса.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

# Теперь код — только то, что реально нужно Inference: общая логика (common),
# файловый регистр моделей (registry) и сам сервис.
COPY src/__init__.py ./src/__init__.py
COPY src/common ./src/common
COPY src/registry ./src/registry
COPY src/inference ./src/inference

EXPOSE 8000

# --workers 1 намеренно: у каждого доп. процесса uvicorn была бы своя копия
# модели в памяти, и /model/switch обновил бы только один из них.
CMD ["uv", "run", "uvicorn", "src.inference.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
