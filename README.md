# How to run

## Запуск сервисов:

`docker-compose up -d --build`

## Миграция таблиц:

`pgmigrate migrate -d db -t latest`

## Вставка тестовых данных в таблицы (если надо):

`bash insert_data.sh`

Можно переопределить параметры:

`CONTAINER_NAME=hse-masters-backend-postgres-1 USERS_COUNT=100 ITEMS_COUNT=5000 ACCOUNTS_COUNT=50 bash insert_data.sh`

## Команды для ручной проверки API

### 1) Логин и сохранение cookie

```bash
curl -i -c cookies.txt -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{"login":"test_user_1","password":"pass123"}'
```

### 2) Проверка, что без авторизации доступ запрещен

```bash
curl -i -X POST "http://localhost:8000/predict/async_predict?item_id=1"
```

### 3) Создание async moderation задачи (с cookie)

```bash
curl -i -b cookies.txt -X POST "http://localhost:8000/predict/async_predict?item_id=1"
```

### 4) Нагрузочная отправка задач (10 параллельных потоков)

```bash
seq 1000 | xargs -I {} -P 10 curl -s -o /dev/null -w "%{http_code}\n" \
  -b cookies.txt -X POST "http://localhost:8000/predict/async_predict?item_id={}"
```

### 5) Проверка результата модерации

```bash
curl -i -b cookies.txt -X GET "http://localhost:8000/predict/moderation_result/1"
```

### 6) Закрытие объявления

```bash
curl -i -b cookies.txt -X POST "http://localhost:8000/predict/close?item_id=1"
```

### 7) Проверка blocked-аккаунта

```bash
curl -i -X POST "http://localhost:8000/login" \
  -H "Content-Type: application/json" \
  -d '{"login":"blocked_user","password":"blocked123"}'
```

### 8) Просмотр логов воркера

```bash
docker-compose logs -f worker
```

### 9) Быстрая проверка числа задач в БД

```bash
docker exec -it hse-masters-backend-postgres-1 psql -U postgres -d hw \
  -c "SELECT status, count(*) FROM moderation_results GROUP BY status ORDER BY status;"
```

### В итоге, получили запущенные сервисы:
- server
- worker
- postgres
- redpanda
- redis
- prometheus
- grafana