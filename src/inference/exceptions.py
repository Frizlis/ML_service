from src.common.exceptions import AppError


class ModelNotLoadedError(AppError):
    """Ни одна версия модели ещё не загружена в Inference (регистр пуст)."""
