"""
Единая точка конфигурации проекта.

По требованиям проекта все настройки должны храниться в .env — здесь мы
описываем их как типизированную pydantic-модель (BaseSettings), а не читаем
os.environ руками. Плюсы такого подхода для учебного проекта:
  - pydantic сам провалидирует типы (например, TRAIN_BATCH_SIZE не должен
    оказаться строкой) и упадёт с понятной ошибкой ещё на старте контейнера,
    а не где-то в середине обучения;
  - все параметры видны в одном месте, а не разбросаны по os.getenv() вызовам
    по всему коду;
  - Inference и Train — разные процессы/контейнеры, но читают один и тот же
    .env через docker-compose (env_file), поэтому переменные специфичные для
    одного сервиса помечены префиксом (INFERENCE_*, TRAIN_*), а общие для
    обоих сервисов (например, путь к регистру моделей) — без префикса.

Settings создаётся один раз при импорте модуля (см. `settings = Settings()`
внизу файла) и переиспользуется во всём проекте как синглтон — это осознанно:
конфигурация не должна меняться на лету во время работы процесса.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Общие для Inference и Train ---
    log_level: str = "INFO"
    model_registry_path: Path = Path("model_registry")

    # --- Inference-сервис ---
    inference_host: str = "0.0.0.0"
    inference_port: int = 8000

    # --- Train-пайплайн ---
    train_data_raw_path: Path = Path("data/raw/News_Category_Dataset_v3.json")
    train_data_processed_path: Path = Path("data/processed")
    train_base_model: str = "roberta-base"
    train_max_len: int = 500
    train_batch_size: int = 32
    train_accum_steps: int = 4
    train_epochs: int = 100
    train_early_stop: int = 15
    train_lr: float = 2e-5
    train_warmup_frac: float = 0.1
    train_weight_decay: float = 0.01
    # True — замораживает энкодер и обучает только классификационную голову
    # (быстрее, меньше VRAM, но обычно слабее качество, чем full fine-tuning).
    train_freeze_base: bool = True
    # LR для режима train_freeze_base=True — на порядки выше, чем train_lr:
    # обучается только голова (сотни тысяч параметров, не 125M), риска
    # катастрофического забывания предобученных весов нет (энкодер не
    # трогаем), поэтому и учиться можно/нужно быстрее. train_lr, подобранный
    # под full fine-tuning всей сети, для одной головы слишком мал — на
    # практике loss почти не двигается за десятки эпох.
    train_freeze_lr: float = 1e-3
    # Метрика, по которой принимается решение о деплое новой модели. Единая
    # метрика используется и для сравнения с абсолютным порогом, и для
    # сравнения с текущей активной моделью — иначе легко получить рассинхрон
    # (например, порог по accuracy, а сравнение с активной — по f1).
    train_validation_metric: str = "f1_weighted"
    train_min_metric_threshold: float = 0.55
    # Имя сервиса inference в docker-сети compose — используется Train'ом для
    # HTTP-вызова эндпоинта смены модели после успешного деплоя.
    train_inference_url: str = "http://inference:8000"
    train_deploy_timeout_seconds: float = 10.0
    train_seed: int = 42


# Синглтон конфигурации — импортируется как `from src.common.config import settings`.
settings = Settings()
