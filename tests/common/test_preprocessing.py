"""
Юнит-тесты для src/common/preprocessing.py.

Это чистые функции без сторонних зависимостей (кроме токенизатора, который
здесь подменяется фейком, чтобы тест не тянул модель из интернета) — идеальные
кандидаты для unit-тестирования: дёшево запускать, легко ловят регрессии.
"""
import pytest

from src.common.exceptions import EmptyTextError
from src.common.preprocessing import (
    build_input_text,
    is_valid_text,
    normalize_text,
    tokenize_texts,
    validate_and_build_input_text,
)


def test_normalize_text_replaces_curly_apostrophe():
    assert normalize_text("don’t") == "don't"


def test_normalize_text_strips_surrounding_whitespace():
    assert normalize_text("  hello  ") == "hello"


def test_build_input_text_joins_with_literal_sep_token():
    result = build_input_text("Headline", "Short description")
    assert result == "Headline [SEP] Short description"


def test_build_input_text_normalizes_both_parts():
    result = build_input_text("don’t stop", "believ’n")
    assert result == "don't stop [SEP] believ'n"


@pytest.mark.parametrize(
    "headline,short_description,expected",
    [
        ("Headline", "Description", True),
        ("", "Description", False),
        ("Headline", "", False),
        ("   ", "Description", False),
        ("Headline", "   ", False),
    ],
)
def test_is_valid_text(headline, short_description, expected):
    assert is_valid_text(headline, short_description) is expected


def test_validate_and_build_input_text_returns_input_text_when_valid():
    result = validate_and_build_input_text("Headline", "Description")
    assert result == "Headline [SEP] Description"


def test_validate_and_build_input_text_raises_on_empty_headline():
    with pytest.raises(EmptyTextError):
        validate_and_build_input_text("", "Description")


def test_validate_and_build_input_text_raises_on_empty_short_description():
    with pytest.raises(EmptyTextError):
        validate_and_build_input_text("Headline", "")


class _FakeTokenizer:
    """Минимальный дублёр PreTrainedTokenizer — фиксирует, с какими аргументами
    его вызвали, не скачивая реальную модель токенизатора из интернета."""

    def __call__(self, texts, max_length, padding, truncation, return_tensors):
        self.last_call_kwargs = {
            "texts": texts,
            "max_length": max_length,
            "padding": padding,
            "truncation": truncation,
            "return_tensors": return_tensors,
        }
        return {"input_ids": [[0] * max_length for _ in texts]}


def test_tokenize_texts_uses_expected_params_and_default_max_len():
    tokenizer = _FakeTokenizer()
    tokenize_texts(tokenizer, ["a", "b"])
    assert tokenizer.last_call_kwargs["max_length"] == 500  # settings.train_max_len по умолчанию
    assert tokenizer.last_call_kwargs["padding"] == "max_length"
    assert tokenizer.last_call_kwargs["truncation"] is True
    assert tokenizer.last_call_kwargs["return_tensors"] == "pt"


def test_tokenize_texts_respects_explicit_max_len():
    tokenizer = _FakeTokenizer()
    tokenize_texts(tokenizer, ["a"], max_len=128)
    assert tokenizer.last_call_kwargs["max_length"] == 128
