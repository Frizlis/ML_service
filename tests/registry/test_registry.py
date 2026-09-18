"""
Юнит-тесты для src/registry/registry.py.

Каждый тест создаёт ModelRegistry поверх pytest-фикстуры tmp_path — так
тесты не трогают реальный model_registry/ проекта и полностью изолированы
друг от друга (в отличие от использования глобального синглмона `registry`).
"""
import pytest

from src.registry.exceptions import ModelNotFoundError, NoActiveModelError
from src.registry.registry import ModelRegistry


@pytest.fixture
def registry(tmp_path):
    return ModelRegistry(root_path=tmp_path)


def test_fresh_registry_has_no_versions(registry):
    assert registry.list_versions() == []


def test_fresh_registry_raises_on_get_active(registry):
    with pytest.raises(NoActiveModelError):
        registry.get_active()


def test_register_adds_version_as_archived(registry):
    entry = registry.register("v1", {"f1_weighted": 0.6})
    assert entry.status == "archived"
    assert entry.metrics == {"f1_weighted": 0.6}

    versions = registry.list_versions()
    assert len(versions) == 1
    assert versions[0].version == "v1"


def test_activate_sets_active_version(registry):
    registry.register("v1", {"f1_weighted": 0.6})
    registry.activate("v1")

    active = registry.get_active()
    assert active.version == "v1"
    assert active.status == "active"


def test_activate_unknown_version_raises(registry):
    with pytest.raises(ModelNotFoundError):
        registry.activate("does-not-exist")


def test_activate_archives_previous_active_version(registry):
    registry.register("v1", {"f1_weighted": 0.6})
    registry.register("v2", {"f1_weighted": 0.7})

    registry.activate("v1")
    registry.activate("v2")

    assert registry.get_active().version == "v2"
    assert registry.get_version("v1").status == "archived"
    assert registry.get_version("v2").status == "active"


def test_get_version_unknown_raises(registry):
    with pytest.raises(ModelNotFoundError):
        registry.get_version("does-not-exist")


def test_get_active_model_dir_points_inside_models_dir(registry, tmp_path):
    registry.register("v1", {"f1_weighted": 0.6})
    registry.activate("v1")

    assert registry.get_active_model_dir() == tmp_path / "models" / "v1"


def test_list_versions_sorted_newest_first(registry):
    registry.register("20260101T000000", {"f1_weighted": 0.5})
    registry.register("20260601T000000", {"f1_weighted": 0.6})
    registry.register("20260301T000000", {"f1_weighted": 0.55})

    versions = [entry.version for entry in registry.list_versions()]
    assert versions == ["20260601T000000", "20260301T000000", "20260101T000000"]


def test_index_persists_across_registry_instances(tmp_path):
    first = ModelRegistry(root_path=tmp_path)
    first.register("v1", {"f1_weighted": 0.6})
    first.activate("v1")

    second = ModelRegistry(root_path=tmp_path)
    active = second.get_active()
    assert active.version == "v1"
    assert active.status == "active"


def test_register_does_not_change_active_version(registry):
    registry.register("v1", {"f1_weighted": 0.6})
    registry.activate("v1")

    registry.register("v2", {"f1_weighted": 0.4})  # хуже метрика — деплоить не будем

    assert registry.get_active().version == "v1"
    assert registry.get_version("v2").status == "archived"
