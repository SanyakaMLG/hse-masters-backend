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
        await conn.execute(
            """
            CREATE TABLE IF NOT EXISTS schema_migrations (
                version TEXT PRIMARY KEY,
                applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        applied_rows = await conn.fetch("SELECT version FROM schema_migrations")
        applied_versions = {row["version"] for row in applied_rows}

        for migration_path in sorted(migrations_dir.glob("V*.sql")):
            if migration_path.name in applied_versions:
                print(f"Skipped {migration_path.name}")
                continue

            migration_sql = migration_path.read_text(encoding="utf-8")
            async with conn.transaction():
                await conn.execute(migration_sql)
                await conn.execute(
                    "INSERT INTO schema_migrations (version) VALUES ($1)",
                    migration_path.name,
                )
            print(f"Applied {migration_path.name}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(apply_migrations())
