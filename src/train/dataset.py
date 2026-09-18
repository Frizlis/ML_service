"""
Загрузка и подготовка обучающих данных: сырой JSONL -> очищенный DataFrame ->
time-based split -> токенизированные torch Dataset'ы.

Это train-специфичная часть препроцессинга (в отличие от src/common/preprocessing.py,
которую переиспользует ещё и inference): дедупликация, разбиение на выборки,
LabelEncoder и class weights нужны только на этапе обучения — inference на вход
получает уже готовый (не "сырой") текст одного запроса, ему сплит и веса классов
не нужны.

Логика ниже переносит уже провалидированный метриками пайплайн из референсного
ноутбука (data/raw/News_Category_Dataset_v3.json, roberta-base, val accuracy ~0.63).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset

from src.common.logging_config import get_logger
from src.common.preprocessing import build_input_text, is_valid_text, tokenize_texts

logger = get_logger(__name__)


class NewsDataset(Dataset):
    """torch Dataset над уже токенизированными текстами и закодированными метками."""

    def __init__(self, encodings, labels: np.ndarray):
        self.input_ids = encodings["input_ids"]
        self.attention_mask = encodings["attention_mask"]
        self.labels = torch.as_tensor(labels, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def load_raw_dataframe(path: Path) -> pd.DataFrame:
    """Читает датасет в формате JSON Lines (по одному JSON-объекту на строку)."""
    logger.info("loading_raw_data", extra={"path": str(path)})
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    df = pd.DataFrame.from_records(records)
    logger.info("raw_data_loaded", extra={"rows": len(df)})
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Дедупликация + фильтрация пустых текстов + сборка input_text + сортировка
    по дате (нужна для последующего time-based split без утечки данных из
    будущего в прошлое).
    """
    rows_before = len(df)

    df = df.drop_duplicates()
    df = df.drop_duplicates(subset=["headline"])
    df = df.drop_duplicates(subset=["short_description"])

    valid_mask = df.apply(
        lambda row: is_valid_text(row["headline"], row["short_description"]), axis=1
    )
    df = df[valid_mask].copy()

    df["input_text"] = df.apply(
        lambda row: build_input_text(row["headline"], row["short_description"]), axis=1
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    logger.info(
        "data_cleaned", extra={"rows_before": rows_before, "rows_after": len(df)}
    )
    return df


def time_based_split(
    df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Хронологическое разбиение (df уже отсортирован по возрастанию даты):
    train — самые старые записи, val — следующие по времени, test — самые
    свежие. В отличие от случайного split, это не даёт модели "подсматривать"
    в будущее относительно того, на чём её потом реально тестируют.
    """
    if not (0 < train_frac and 0 < val_frac and train_frac + val_frac < 1):
        raise ValueError(f"invalid fractions: {train_frac=}, {val_frac=}")

    train_df, val_df, test_df = [], [], []
    for cat, g in df.groupby("category"):
        if len(g) < 3:
            logger.warning("class_too_small_for_split", extra={"category": cat, "len": len(g)})
        n = int(len(g) * train_frac)
        m = int(len(g) * (train_frac + val_frac))
        train_df.append(g.iloc[:n])
        val_df.append(g.iloc[n:m])
        test_df.append(g.iloc[m:])

    train_df = pd.concat(train_df, ignore_index=True)
    val_df = pd.concat(val_df, ignore_index=True)
    test_df = pd.concat(test_df, ignore_index=True)

    logger.info(
        "data_split",
        extra={"train": len(train_df), "val": len(val_df), "test": len(test_df)},
    )
    return train_df, val_df, test_df


class EncodedSplits(NamedTuple):
    encoder: LabelEncoder
    train_labels: np.ndarray
    val_df: pd.DataFrame
    val_labels: np.ndarray
    test_df: pd.DataFrame
    test_labels: np.ndarray


def _drop_unseen_categories(df: pd.DataFrame, known_classes: set[str]) -> pd.DataFrame:
    mask = df["category"].isin(known_classes)
    dropped = int((~mask).sum())
    if dropped:
        unknown_categories = set(df[~mask]["category"].unique())
        logger.warning(
            "dropping_rows_with_unseen_category",
            extra={"dropped_rows": dropped, "total_rows": len(df),
                   "unknown_categories": unknown_categories},
        )
    return df[mask].reset_index(drop=True)


def encode_labels(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> EncodedSplits:
    """
    LabelEncoder обучается ТОЛЬКО на train (иначе классы, встречающиеся лишь
    в val/test, незаметно "утекли" бы в разметку модели на этапе обучения).

    Строки val/test с категорией, отсутствующей в train, отбрасываются с
    явным предупреждением в лог, а не роняют весь пайплайн исключением:
    такая строка всё равно не несёт полезного сигнала для модели, которая
    эту категорию на обучении не видела. На полном датасете (186k+ строк,
    42 класса) такое практически не встречается — это защита от краевого
    случая на маленьких/нерепрезентативных срезах данных (например, smoke-тест
    на первых N строках датасета).
    """
    encoder = LabelEncoder()
    train_labels = encoder.fit_transform(train_df["category"])
    known_classes = set(encoder.classes_)

    val_df = _drop_unseen_categories(val_df, known_classes)
    test_df = _drop_unseen_categories(test_df, known_classes)

    val_labels = encoder.transform(val_df["category"])
    test_labels = encoder.transform(test_df["category"])

    return EncodedSplits(encoder, train_labels, val_df, val_labels, test_df, test_labels)


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Датасет сильно несбалансирован (например, POLITICS на два порядка больше
    LATINO VOICES) — без весов классов модель научится игнорировать редкие
    категории. weight[c] = N / (num_classes * count[c]) — чем реже класс, тем
    больше его вес в CrossEntropyLoss.
    """
    _, counts = np.unique(labels, return_counts=True)
    weights = [len(labels) / (len(counts) * count) for count in counts]
    return torch.tensor(weights, dtype=torch.float)


def build_dataset(df: pd.DataFrame, labels: np.ndarray, tokenizer, max_len: int) -> NewsDataset:
    """Токенизирует уже собранный input_text через общую (common) функцию токенизации."""
    encodings = tokenize_texts(tokenizer, df["input_text"].tolist(), max_len)
    return NewsDataset(encodings, labels)
