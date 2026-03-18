BEGIN;

ALTER TABLE account
ADD CONSTRAINT account_login_unique UNIQUE (login);

COMMIT;
