"""
Юнит-тесты для src/train/dataset.py — очистка данных, time-based split,
кодирование меток, class weights. Всё на маленьких синтетических DataFrame,
без реального датасета и без сети.
"""
import numpy as np
import pandas as pd
import pytest

from src.train.dataset import (
    clean_dataframe,
    compute_class_weights,
    encode_labels,
    time_based_split,
)


def _make_raw_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame.from_records(rows)


def test_clean_dataframe_drops_full_duplicates():
    df = _make_raw_df(
        [
            {"headline": "A", "short_description": "a-desc", "category": "X", "date": "2020-01-01"},
            {"headline": "A", "short_description": "a-desc", "category": "X", "date": "2020-01-01"},
        ]
    )
    cleaned = clean_dataframe(df)
    assert len(cleaned) == 1


def test_clean_dataframe_drops_rows_with_empty_headline_or_description():
    df = _make_raw_df(
        [
            {"headline": "A", "short_description": "a-desc", "category": "X", "date": "2020-01-01"},
            {"headline": "", "short_description": "b-desc", "category": "Y", "date": "2020-01-02"},
            {"headline": "C", "short_description": "", "category": "Z", "date": "2020-01-03"},
        ]
    )
    cleaned = clean_dataframe(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["headline"] == "A"


def test_clean_dataframe_builds_input_text_and_sorts_by_date():
    df = _make_raw_df(
        [
            {"headline": "Later", "short_description": "later-desc", "category": "X", "date": "2020-02-01"},
            {"headline": "Earlier", "short_description": "earlier-desc", "category": "Y", "date": "2020-01-01"},
        ]
    )
    cleaned = clean_dataframe(df)
    assert list(cleaned["headline"]) == ["Earlier", "Later"]
    assert cleaned.iloc[0]["input_text"] == "Earlier [SEP] earlier-desc"


def test_time_based_split_sizes_and_no_overlap():
    df = _make_raw_df(
        [
            {"headline": f"H{i}", "short_description": f"D{i}", "category": "X", "date": f"2020-01-{i + 1:02d}"}
            for i in range(10)
        ]
    )
    train_df, val_df, test_df = time_based_split(df, train_frac=0.6, val_frac=0.2)

    assert len(train_df) == 6
    assert len(val_df) == 2
    assert len(test_df) == 2

    all_headlines = list(train_df["headline"]) + list(val_df["headline"]) + list(test_df["headline"])
    assert sorted(all_headlines) == sorted(df["headline"])
    assert len(set(all_headlines)) == len(all_headlines)  # без пересечений


def test_time_based_split_preserves_chronological_order():
    df = _make_raw_df(
        [
            {"headline": f"H{i}", "short_description": f"D{i}", "category": "X", "date": f"2020-01-{i + 1:02d}"}
            for i in range(10)
        ]
    )
    train_df, val_df, test_df = time_based_split(df, train_frac=0.6, val_frac=0.2)

    # train — самые старые записи, test — самые свежие.
    assert train_df["headline"].tolist() == [f"H{i}" for i in range(6)]
    assert test_df["headline"].tolist() == [f"H{i}" for i in range(8, 10)]


def test_encode_labels_fits_only_on_train():
    train_df = _make_raw_df([{"category": "A"}, {"category": "B"}])
    val_df = _make_raw_df([{"category": "A"}])
    test_df = _make_raw_df([{"category": "B"}])

    encoded = encode_labels(train_df, val_df, test_df)

    assert set(encoded.encoder.classes_) == {"A", "B"}
    assert len(encoded.train_labels) == 2
    assert len(encoded.val_labels) == 1
    assert len(encoded.test_labels) == 1


def test_encode_labels_drops_rows_with_unseen_category_instead_of_raising():
    train_df = _make_raw_df([{"category": "A"}])
    val_df = _make_raw_df([{"category": "A"}, {"category": "UNSEEN"}])
    test_df = _make_raw_df([{"category": "A"}])

    encoded = encode_labels(train_df, val_df, test_df)

    # Строка с "UNSEEN" тихо не роняет пайплайн, а просто исключается —
    # и df, и labels остаются согласованы по длине.
    assert len(encoded.val_df) == 1
    assert len(encoded.val_labels) == 1
    assert encoded.val_df.iloc[0]["category"] == "A"


def test_compute_class_weights_gives_higher_weight_to_rare_class():
    # Класс 0 встречается 8 раз, класс 1 — только 2 раза из 10.
    labels = np.array([0] * 8 + [1] * 2)
    weights = compute_class_weights(labels)

    assert weights[1] > weights[0]


def test_compute_class_weights_equal_for_balanced_classes():
    labels = np.array([0, 0, 1, 1])
    weights = compute_class_weights(labels)

    assert weights[0] == pytest.approx(weights[1])
