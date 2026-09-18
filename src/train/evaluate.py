"""
Оценка качества модели и validation gate — решение, можно ли деплоить новую
версию поверх текущей активной.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.common.logging_config import get_logger

logger = get_logger(__name__)


@torch.no_grad()
def compute_predictions(
    model, dataset, device: torch.device, batch_size: int = 32
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    dl = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_preds, all_labels = [], []
    for batch in dl:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        all_preds.append(logits.argmax(dim=-1).cpu().numpy())
        all_labels.append(batch["labels"].numpy())
    return np.concatenate(all_preds), np.concatenate(all_labels)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """
    weighted F1 — основная метрика для 42 несбалансированных классов (accuracy
    легко "накрутить", просто угадывая доминирующий класс POLITICS). accuracy
    считаем тоже — просто для наглядности в логах/метриках версии.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
    }


def validation_gate(
    new_metrics: dict[str, float],
    *,
    metric_name: str,
    min_threshold: float,
    active_metrics: dict[str, float] | None,
) -> tuple[bool, str]:
    """
    Новая модель проходит гейт, только если ОБА условия выполнены:
      1) metric_name >= абсолютного порога из .env (TRAIN_MIN_METRIC_THRESHOLD);
      2) metric_name не хуже метрики текущей активной модели (если активная
         модель уже есть — на самом первом обучении сравнивать не с чем, и
         это условие автоматически пропускается).

    Единая метрика используется в обоих сравнениях осознанно: порог по одной
    метрике и сравнение с активной моделью по другой легко приводят к
    ситуации, когда обе проверки формально пройдены, а модель на деле хуже.
    """
    value = new_metrics.get(metric_name)
    if value is None:
        return False, f"Метрика '{metric_name}' отсутствует в результатах оценки"

    if value < min_threshold:
        return False, f"{metric_name}={value:.4f} ниже порога {min_threshold:.4f}"

    if active_metrics is not None:
        active_value = active_metrics.get(metric_name)
        if active_value is not None and value < active_value:
            return (
                False,
                f"{metric_name}={value:.4f} хуже текущей активной модели ({active_value:.4f})",
            )

    return (
        True,
        f"{metric_name}={value:.4f} прошла порог {min_threshold:.4f} и сравнение с активной моделью",
    )
