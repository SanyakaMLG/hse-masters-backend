# How to run

## Запуск сервисов:

`docker-compose up -d --build`

## Миграция таблиц:

`pgmigrate migrate -d db -t latest`

## Вставка тестовых данных в таблицы (если надо):

`bash insert_data.sh`

### В итоге, получили запущенные сервисы:
- server
- worker
- postgres
- redpanda