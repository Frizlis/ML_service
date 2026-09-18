"""
Entrypoint train-пайплайна: dataset -> trainer -> evaluate -> deploy.

Запускается как процесс `uv run python -m src.train.main`. По завершении
процесс выходит — это и есть механизм "Train-контейнер выключается по
окончании работы" (см. `docker compose run --rm train` в scripts/).

Важное архитектурное допущение: каждый запуск дообучает модель ЗАНОВО от
базового чекпоинта (settings.train_base_model), а не продолжает дообучение
поверх текущей активной версии. Так проще рассуждать о качестве (нет риска
катастрофического забывания при многократном дообучении) и так же обучалась
модель в референсном ноутбуке. "Берёт новые данные" означает, что при
каждом запуске Train читает файл данных заново — если он обновился между
запусками, это учтётся.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.common.config import settings
from src.common.logging_config import get_logger
from src.registry.exceptions import NoActiveModelError
from src.registry.registry import ModelRegistry, new_version_id
from src.registry.registry import registry as default_registry
from src.train import dataset as ds
from src.train.deploy import deploy_version
from src.train.evaluate import compute_metrics, compute_predictions, validation_gate
from src.train.trainer import train_model

logger = get_logger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_training_pipeline(
    *,
    registry: ModelRegistry | None = None,
    data_path: Path | None = None,
    base_model: str | None = None,
    max_len: int | None = None,
    batch_size: int | None = None,
    accum_steps: int | None = None,
    epochs: int | None = None,
    early_stop: int | None = None,
    lr: float | None = None,
    warmup_frac: float | None = None,
    weight_decay: float | None = None,
    freeze_base: bool | None = None,
    max_rows: int | None = None,
) -> dict:
    """
    Полный пайплайн обучения. Параметры со значением None берутся из
    settings (.env) — прод-запуск вызывается без единого аргумента. Явные
    аргументы (в первую очередь max_rows/epochs) нужны для быстрого
    smoke-теста пайплайна на маленькой выборке, не трогая .env реального
    конфига.
    """
    registry = registry or default_registry
    data_path = data_path or settings.train_data_raw_path
    base_model = base_model or settings.train_base_model
    max_len = max_len or settings.train_max_len
    batch_size = batch_size or settings.train_batch_size
    accum_steps = accum_steps or settings.train_accum_steps
    epochs = epochs if epochs is not None else settings.train_epochs
    early_stop = early_stop if early_stop is not None else settings.train_early_stop
    lr = lr or settings.train_lr
    warmup_frac = warmup_frac if warmup_frac is not None else settings.train_warmup_frac
    weight_decay = weight_decay if weight_decay is not None else settings.train_weight_decay
    freeze_base = freeze_base if freeze_base is not None else settings.train_freeze_base

    _set_seed(settings.train_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("training_pipeline_started", extra={"device": str(device), "base_model": base_model})

    df = ds.load_raw_dataframe(data_path)
    df = ds.clean_dataframe(df)
    if max_rows is not None:
        df = df.iloc[:max_rows].reset_index(drop=True)

    train_df, val_df, test_df = ds.time_based_split(df)
    encoded = ds.encode_labels(train_df, val_df, test_df)
    num_classes = len(encoded.encoder.classes_)
    class_weights = ds.compute_class_weights(encoded.train_labels)

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    train_dataset = ds.build_dataset(train_df, encoded.train_labels, tokenizer, max_len)
    val_dataset = ds.build_dataset(encoded.val_df, encoded.val_labels, tokenizer, max_len)
    test_dataset = ds.build_dataset(encoded.test_df, encoded.test_labels, tokenizer, max_len)

    # id2label/label2id кладём прямо в config.json модели через save_pretrained
    # ниже — отдельный labels.json не заводим, чтобы не иметь два источника
    # истины для одного и того же маппинга.
    id2label = {i: label for i, label in enumerate(encoded.encoder.classes_)}
    label2id = {label: i for i, label in id2label.items()}
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=num_classes,
        id2label=id2label,
        label2id=label2id,
        problem_type="single_label_classification",
        # base_model конфигурируется через .env (TRAIN_BASE_MODEL) и не всегда
        # будет "чистым" MLM-чекпоинтом вроде roberta-base — если у него уже
        # есть classifier-голова другого размера, не роняем пайплайн, а
        # переинициализируем голову под наше число классов.
        ignore_mismatched_sizes=True,
    )
    model
    model.to(device)

    train_model(
        model,
        train_dataset,
        val_dataset,
        class_weights,
        device,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        weight_decay=weight_decay,
        warmup_frac=warmup_frac,
        accum_steps=accum_steps,
        early_stop_patience=early_stop,
        freeze_base=freeze_base,
    )

    preds, trues = compute_predictions(model, test_dataset, device, batch_size=batch_size)
    metrics = compute_metrics(trues, preds)
    logger.info("evaluation_completed", extra=metrics)

    version_id = new_version_id()
    version_dir = registry.version_dir(version_id)
    version_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(version_dir)
    tokenizer.save_pretrained(version_dir)
    registry.register(version_id, metrics)

    try:
        active_metrics = registry.get_active().metrics
    except NoActiveModelError:
        active_metrics = None

    passed, reason = validation_gate(
        metrics,
        metric_name=settings.train_validation_metric,
        min_threshold=settings.train_min_metric_threshold,
        active_metrics=active_metrics,
    )
    logger.info(
        "validation_gate_result",
        extra={"version": version_id, "passed": passed, "reason": reason},
    )

    if passed:
        deploy_version(registry, version_id)
    else:
        logger.warning("model_not_deployed", extra={"version": version_id, "reason": reason})

    return {"version": version_id, "metrics": metrics, "deployed": passed, "reason": reason}


if __name__ == "__main__":
    run_training_pipeline()
