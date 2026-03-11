#!/bin/bash

set -euo pipefail

CONTAINER_NAME="${CONTAINER_NAME:-hse-masters-backend-postgres-1}"
DB_USER="${DB_USER:-postgres}"
DB_NAME="${DB_NAME:-hw}"
USERS_COUNT="${USERS_COUNT:-50}"
ITEMS_COUNT="${ITEMS_COUNT:-1000}"
ACCOUNTS_COUNT="${ACCOUNTS_COUNT:-20}"

if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
  echo "[ERROR] Container '$CONTAINER_NAME' is not running"
  exit 1
fi

echo "[INFO] Waiting for postgres in $CONTAINER_NAME..."
docker exec -i "$CONTAINER_NAME" pg_isready -U "$DB_USER" -d "$DB_NAME" >/dev/null

echo "[INFO] Inserting users (up to $USERS_COUNT)"
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME" <<SQL
INSERT INTO users (id, is_verified_seller)
SELECT
    gs,
    (random() > 0.5)
FROM generate_series(1, $USERS_COUNT) gs
ON CONFLICT (id) DO NOTHING;
SQL

echo "[INFO] Inserting items ($ITEMS_COUNT rows)"
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME" <<SQL
INSERT INTO items (user_id, name, description, category, images_qty)
SELECT
    floor(random() * $USERS_COUNT + 1)::int,
    'Товар №' || gs,
    'Это описание тестового товара под номером ' || gs || '. Оно должно быть достаточно длинным для проверки логики нашей модели модерации.',
    floor(random() * 100 + 1)::int,
    floor(random() * 10)::int
FROM generate_series(1, $ITEMS_COUNT) gs;
SQL

echo "[INFO] Inserting test accounts ($ACCOUNTS_COUNT rows, password=pass123)"
docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME" <<SQL
INSERT INTO account (login, password, is_blocked)
SELECT
    'test_user_' || gs,
    md5('pass123'),
    FALSE
FROM generate_series(1, $ACCOUNTS_COUNT) gs
ON CONFLICT DO NOTHING;
SQL

# Один специально заблокированный аккаунт для проверки 401
# пароль: blocked123

docker exec -i "$CONTAINER_NAME" psql -v ON_ERROR_STOP=1 -U "$DB_USER" -d "$DB_NAME" <<SQL
INSERT INTO account (login, password, is_blocked)
VALUES ('blocked_user', md5('blocked123'), TRUE)
ON CONFLICT DO NOTHING;
SQL

echo "[INFO] Data stats"
docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT count(*) AS total_users FROM users;"
docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT count(*) AS total_items FROM items;"
docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT count(*) AS total_accounts FROM account;"
docker exec -i "$CONTAINER_NAME" psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT count(*) AS blocked_accounts FROM account WHERE is_blocked = TRUE;"

echo "[INFO] Done. Example accounts: test_user_1/pass123, blocked_user/blocked123"