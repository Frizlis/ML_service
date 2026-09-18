#!/usr/bin/env bash
# Запускает Train по требованию.
#
# `docker compose run --rm` (а не `up`) — принципиально: `run --rm`
# гарантирует, что контейнер удалится сам сразу после завершения
# тренировочного процесса. Это и есть механизм "Train выключается по
# окончании работы" из требований проекта — restart-политика в
# docker-compose.yml на `run` не действует, всё делает --rm.
#
# Если Inference уже поднят (docker compose up -d inference), Train сразу
# после успешного деплоя уведомит его по HTTP и новая модель подхватится
# "на лету", без рестарта Inference. Если Inference не поднят — это не
# ошибка: он прочитает активную версию из model_registry/registry.json сам
# при следующем своём запуске.
set -euo pipefail

docker compose run --rm train
