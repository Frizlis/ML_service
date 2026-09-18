# Train-сервис: запускается по требованию (`docker compose run --rm train`,
# см. scripts/train.ps1 / train.sh) и обучает модель на GPU. restart-политика
# в docker-compose.yml здесь не имеет значения для `run` — реальное
# самовыключение контейнера по завершении обеспечивает флаг --rm.
#
# Базовый образ — обычный python:3.11-slim, БЕЗ nvidia/cuda-образа: torch
# ставится через uv с индексом pytorch-cu128 (см. pyproject.toml) и уже
# бандлит нужные CUDA-библиотеки (cuDNN/cuBLAS и т.д.) сам. Доступ к GPU
# докер прокидывает на уровне рантайма (see docker-compose.yml:
# deploy.resources.reservations.devices), тяжёлый devel-образ с CUDA
# toolkit тут не нужен — им бы мы только компилировали что-то из исходников,
# а не наш случай.
FROM python:3.11-slim

RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen

COPY src/__init__.py ./src/__init__.py
COPY src/common ./src/common
COPY src/registry ./src/registry
COPY src/train ./src/train

CMD ["uv", "run", "python", "-m", "src.train.main"]
