from src.common.exceptions import AppError


class ModelNotFoundError(AppError):
    """Запрошенная версия модели отсутствует в регистре."""


class NoActiveModelError(AppError):
    """В регистре ещё нет ни одной активной модели (сервис ещё не обучали)."""
