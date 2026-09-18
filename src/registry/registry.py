"""
Файловое хранилище (регистр) версий модели — реализация требования
"внутреннее хранилище для ML моделей".

Почему отдельный пакет src/registry, а не часть src/common: common отвечает
за общую ML-логику (препроцессинг) — чистые функции без побочных эффектов.
Регистр — стейтфул-компонент с файловым I/O и версионированием, у него
другая природа, и оба сервиса (Train — пишет, Inference — читает) обращаются
к нему как к независимому инфраструктурному модулю.

Формат хранения:
  model_registry/
  ├── registry.json          # индекс: активная версия + метрики всех версий
  └── models/<version_id>/   # артефакты конкретной версии (веса, токенизатор)

Конкурентный доступ: в этом проекте пишет в registry.json только Train
(по требованию, не параллельно сам с собой), Inference — только читает.
Полноценная файловая блокировка (fcntl/portalocker) здесь избыточна;
достаточно атомарной записи (temp-file + os.replace), чтобы обрыв процесса
посреди записи не оставил файл в частично записанном состоянии.
"""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.common.config import settings
from src.common.logging_config import get_logger
from src.registry.exceptions import ModelNotFoundError, NoActiveModelError
from src.registry.schemas import ModelVersionEntry, RegistryIndex

logger = get_logger(__name__)

_INDEX_FILENAME = "registry.json"
_MODELS_DIRNAME = "models"


def new_version_id() -> str:
    """
    Сортируемый по времени идентификатор версии модели (UTC), например
    "20260812T190000". Не используем инкрементальный счётчик — пришлось бы
    отдельно хранить и защищать его от гонок; timestamp самодостаточен и
    сортируется лексикографически так же, как по времени создания.
    """
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")


class ModelRegistry:
    def __init__(self, root_path: Path | None = None):
        # Каталоги на диске создаются лениво (в _save_index/через version_dir
        # у вызывающего кода), а не здесь — конструктор не должен трогать
        # файловую систему просто от создания объекта. Иначе даже сам факт
        # импорта модуля (singleton `registry` ниже создаётся при импорте)
        # создавал бы model_registry/models/ на диске без единой реальной
        # записи — неожиданный побочный эффект, особенно в тестах/CI.
        self.root_path = root_path or settings.model_registry_path
        self.models_path = self.root_path / _MODELS_DIRNAME
        self.index_path = self.root_path / _INDEX_FILENAME

    # --- чтение/запись индекса ---

    def _load_index(self) -> RegistryIndex:
        if not self.index_path.exists():
            return RegistryIndex()
        raw = json.loads(self.index_path.read_text(encoding="utf-8"))
        return RegistryIndex.model_validate(raw)

    def _save_index(self, index: RegistryIndex) -> None:
        self.root_path.mkdir(parents=True, exist_ok=True)
        fd, tmp_path_str = tempfile.mkstemp(
            dir=self.root_path, prefix=".registry_", suffix=".tmp"
        )
        tmp_path = Path(tmp_path_str)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(index.model_dump_json(indent=2))
            os.replace(tmp_path, self.index_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise

    # --- версии ---

    def version_dir(self, version_id: str) -> Path:
        """
        Путь к каталогу артефактов версии. Не проверяет существование —
        используется и для подготовки каталога перед сохранением новой
        версии (train), и для чтения уже существующей (inference).
        """
        return self.models_path / version_id

    def register(self, version_id: str, metrics: dict[str, float]) -> ModelVersionEntry:
        """
        Добавляет версию в индекс со статусом "archived" (ещё не активна).
        Вызывается train-пайплайном ПОСЛЕ того, как артефакты модели уже
        сохранены на диск в version_dir(version_id) — чтобы в индексе
        никогда не оказалось записи без реальных файлов на диске.

        Версия регистрируется независимо от того, пройдёт ли она дальше
        validation gate — это сознательно: история всех попыток обучения
        (включая отклонённые) должна быть видна через list_versions().
        """
        index = self._load_index()
        entry = ModelVersionEntry(
            version=version_id,
            created_at=datetime.now(timezone.utc),
            metrics=metrics,
            status="archived",
        )
        index.versions[version_id] = entry
        self._save_index(index)
        logger.info(
            "model_version_registered", extra={"version": version_id, "metrics": metrics}
        )
        return entry

    def activate(self, version_id: str) -> None:
        """
        Помечает версию как активную — именно её должен использовать
        Inference (читает при своём старте, либо получает по HTTP-вызову
        /model/switch сразу после деплоя).
        """
        index = self._load_index()
        if version_id not in index.versions:
            raise ModelNotFoundError(f"Версия модели '{version_id}' не найдена в регистре")

        previous_active = index.active_version
        if previous_active and previous_active in index.versions:
            index.versions[previous_active].status = "archived"

        index.versions[version_id].status = "active"
        index.active_version = version_id
        self._save_index(index)
        logger.info(
            "model_version_activated",
            extra={"version": version_id, "previous_version": previous_active},
        )

    def get_active(self) -> ModelVersionEntry:
        index = self._load_index()
        if index.active_version is None:
            raise NoActiveModelError("В регистре ещё нет ни одной активной модели")
        return index.versions[index.active_version]

    def get_active_model_dir(self) -> Path:
        return self.version_dir(self.get_active().version)

    def get_version(self, version_id: str) -> ModelVersionEntry:
        index = self._load_index()
        if version_id not in index.versions:
            raise ModelNotFoundError(f"Версия модели '{version_id}' не найдена в регистре")
        return index.versions[version_id]

    def list_versions(self) -> list[ModelVersionEntry]:
        """Все версии, от новой к старой (version_id сортируется лексикографически = по времени)."""
        index = self._load_index()
        return sorted(index.versions.values(), key=lambda entry: entry.version, reverse=True)


# Синглтон, аналогично src.common.config.settings — оба сервиса работают
# с одним и тем же регистром на диске (смонтированным как volume в compose).
registry = ModelRegistry()
