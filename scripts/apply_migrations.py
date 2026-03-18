import asyncio
import os
from pathlib import Path

import asyncpg

DEFAULT_DATABASE_URL = "postgresql://postgres:postgres@localhost:5435/hw"


async def apply_migrations() -> None:
    base_dir = Path(__file__).resolve().parent.parent
    migrations_dir = base_dir / "db" / "migrations"
    database_url = os.getenv("DATABASE_URL", DEFAULT_DATABASE_URL)

    conn = await asyncpg.connect(database_url)
    try:
        for migration_path in sorted(migrations_dir.glob("V*.sql")):
            migration_sql = migration_path.read_text(encoding="utf-8")
            await conn.execute(migration_sql)
            print(f"Applied {migration_path.name}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(apply_migrations())
