"""
Общая логика подготовки текста — используется и Train (обучение), и Inference
(предсказание). Именно поэтому она вынесена в common, а не продублирована в
train/ и inference/ по отдельности: если препроцессинг разъедется между двумя
сервисами хотя бы в одной детали (например, inference не будет знать про
литеральный разделитель "[SEP]"), модель на проде будет получать вход,
отличающийся от того, на чём она обучалась, и предсказания станут мусором.

Разделитель между headline и short_description — намеренно обычная подстрока
"[SEP]", а не спецтокен: у roberta-base нет токена [SEP] в словаре (это токен
BERT-семейства), модель просто выучивает эти символы как обычные сабворды.
Это осознанное упрощение, унаследованное из референсного ноутбука (там же и
провалидированное метриками: val accuracy ~0.63) — переход на "честную" пару
последовательностей через tokenizer(text, text_pair=...) — возможное будущее
улучшение, не тема этой итерации.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from src.common.config import settings
from src.common.exceptions import EmptyTextError
from src.common.logging_config import get_logger

if TYPE_CHECKING:
    from transformers import BatchEncoding, PreTrainedTokenizerBase

logger = get_logger(__name__)

# В исходном датасете встречается типографская кавычка вместо обычного
# ASCII-апострофа — модели проще и стабильнее работать с одним вариантом.
_CURLY_APOSTROPHE = "’"
SEP_TOKEN = " [SEP] "


def normalize_text(text: str) -> str:
    """Убирает типографскую кавычку и обрезает пробелы по краям строки."""
    return text.replace(_CURLY_APOSTROPHE, "'").strip()


def build_input_text(headline: str, short_description: str) -> str:
    """
    Собирает единый текст для модели из заголовка и краткого описания —
    ровно так, как модель обучается (см. docstring модуля).
    """
    return f"{normalize_text(headline)}{SEP_TOKEN}{normalize_text(short_description)}"


def is_valid_text(headline: str, short_description: str) -> bool:
    """
    Строка непригодна для обучения/предсказания, если после очистки заголовок
    или описание оказались пустыми — на таких примерах модель либо не может
    ничему научиться (train), либо предсказание не будет иметь смысла (inference).
    """
    return bool(normalize_text(headline)) and bool(normalize_text(short_description))


def validate_and_build_input_text(headline: str, short_description: str) -> str:
    """
    Используется в inference: та же валидация, что применяется к обучающим
    данным, но с явной ошибкой вместо тихого отбрасывания строки — на входе
    в /predict должно быть понятно, почему запрос отклонён, а не получить
    предсказание на пустом тексте.
    """
    if not is_valid_text(headline, short_description):
        logger.warning(
            "empty_text_after_normalization",
            extra={
                "headline_len": len(headline or ""),
                "short_description_len": len(short_description or ""),
            },
        )
        raise EmptyTextError(
            "headline и short_description не должны быть пустыми после нормализации"
        )
    return build_input_text(headline, short_description)


def tokenize_texts(
    tokenizer: "PreTrainedTokenizerBase",
    texts: list[str],
    max_len: int | None = None,
) -> "BatchEncoding":
    """
    Единая точка токенизации для train (батчами) и inference (по одному
    тексту за раз) — совпадающие max_length/padding/truncation критичны,
    иначе распределение входа при инференсе разъедется с тем, что модель
    видела на обучении.
    """
    max_len = max_len or settings.train_max_len
    logger.debug("tokenizing_texts", extra={"count": len(texts), "max_len": max_len})
    return tokenizer(
        texts,
        max_length=max_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
