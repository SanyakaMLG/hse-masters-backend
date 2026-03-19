BEGIN;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'account_login_unique'
    ) THEN
        ALTER TABLE account
        ADD CONSTRAINT account_login_unique UNIQUE (login);
    END IF;
END $$;

COMMIT;
