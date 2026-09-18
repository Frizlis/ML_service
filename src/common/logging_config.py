"""
Фабрика логгеров для всего проекта.

Требование чек-листа: логирование в основных функциях через logger, никаких
print(). Почему это важно не только "для галочки":
  - print() пишет напрямую в stdout без уровней важности (DEBUG/INFO/ERROR) —
    нельзя быстро отфильтровать шум от реальных ошибок;
  - в контейнере Docker логи собирает docker/оркестратор именно из
    stdout/stderr процесса — но если это неструктурированный текст, его
    невозможно нормально парсить и агрегировать (например, в Grafana Loki
    или ELK). JSON-формат (через python-json-logger) решает эту проблему:
    каждая строка лога — валидный JSON-объект с полями asctime/name/level/
    message и любыми дополнительными полями (extra=...).

Используем один и тот же формат и для Inference, и для Train — это тоже
часть "общей логики", хоть и не препроцессинга, а сервисной инфраструктуры.
"""
import logging
import sys

from pythonjsonlogger import json as jsonlogger

from src.common.config import settings

# Флаг гарантирует, что root-логгер настраивается один раз за жизнь процесса,
# даже если get_logger() вызовут из десятка разных модулей при импорте.
_configured = False


def _configure_root_logger() -> None:
    global _configured
    if _configured:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = jsonlogger.JsonFormatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(settings.log_level)
    # Заменяем хендлеры целиком, а не добавляем — иначе при повторном вызове
    # (например, в тестах, где модуль может переимпортироваться) в root
    # логгере накапливались бы дублирующиеся хендлеры и каждая строка лога
    # печаталась бы несколько раз.
    root.handlers = [handler]

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """
    Возвращает именованный логгер (обычно `get_logger(__name__)`), гарантируя,
    что root-логгер уже сконфигурирован под JSON-вывод.
    """
    _configure_root_logger()
    return logging.getLogger(name)
