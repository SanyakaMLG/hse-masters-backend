#!/bin/bash

CONTAINER_NAME="hse-masters-backend-postgres-1"
DB_USER="postgres"
DB_NAME="hw"

docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME <<EOF
INSERT INTO users (id, is_verified_seller)
SELECT 
    gs, 
    (random() > 0.5) -- 50/50 шанс верификации
FROM generate_series(1, 50) gs
ON CONFLICT (id) DO NOTHING;
EOF

docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME <<EOF
INSERT INTO items (user_id, name, description, category, images_qty)
SELECT 
    floor(random() * 50 + 1)::int, -- случайный user_id от 1 до 50
    'Товар №' || gs, 
    'Это описание тестового товара под номером ' || gs || '. Оно должно быть достаточно длинным для проверки логики нашей модели модерации.', 
    floor(random() * 100 + 1)::int, -- случайная категория
    floor(random() * 10)::int      -- случайное кол-во фото от 0 до 15
FROM generate_series(1, 1000) gs;
EOF

docker exec -i $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -c "SELECT count(*) as total_items FROM items;"