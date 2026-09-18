"""
Юнит-тесты для src/train/evaluate.py — в первую очередь validation_gate,
критичная бизнес-логика (ошибка здесь = деплой плохой модели в прод).
"""
import numpy as np
import pytest

from src.train.evaluate import compute_metrics, validation_gate


def test_compute_metrics_perfect_predictions():
    y_true = np.array([0, 1, 2, 0, 1])
    y_pred = np.array([0, 1, 2, 0, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["f1_weighted"] == 1.0


def test_compute_metrics_partial_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 1, 1, 1])
    metrics = compute_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert 0.0 < metrics["f1_weighted"] < 1.0


# --- validation_gate: все значимые комбинации условий ---


def test_gate_passes_threshold_and_no_active_model():
    passed, reason = validation_gate(
        {"f1_weighted": 0.6},
        metric_name="f1_weighted",
        min_threshold=0.5,
        active_metrics=None,
    )
    assert passed is True
    assert "0.6" in reason


def test_gate_fails_below_threshold():
    passed, reason = validation_gate(
        {"f1_weighted": 0.4},
        metric_name="f1_weighted",
        min_threshold=0.5,
        active_metrics=None,
    )
    assert passed is False
    assert "ниже порога" in reason


def test_gate_fails_when_worse_than_active_model():
    passed, reason = validation_gate(
        {"f1_weighted": 0.55},
        metric_name="f1_weighted",
        min_threshold=0.5,
        active_metrics={"f1_weighted": 0.6},
    )
    assert passed is False
    assert "активной" in reason


def test_gate_passes_when_better_than_active_and_above_threshold():
    passed, reason = validation_gate(
        {"f1_weighted": 0.65},
        metric_name="f1_weighted",
        min_threshold=0.5,
        active_metrics={"f1_weighted": 0.6},
    )
    assert passed is True


def test_gate_passes_when_equal_to_active_model():
    # "не хуже" включает равенство — модель с той же метрикой не должна
    # отклоняться только из-за того, что не строго лучше.
    passed, _ = validation_gate(
        {"f1_weighted": 0.6},
        metric_name="f1_weighted",
        min_threshold=0.5,
        active_metrics={"f1_weighted": 0.6},
    )
    assert passed is True


def test_gate_fails_when_metric_missing_from_results():
    passed, reason = validation_gate(
        {"accuracy": 0.9},
        metric_name="f1_weighted",
        min_threshold=0.5,
        active_metrics=None,
    )
    assert passed is False
    assert "отсутствует" in reason
