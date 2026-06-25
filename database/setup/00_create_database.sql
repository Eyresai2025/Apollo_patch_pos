-- Run this from pgAdmin Query Tool while connected as the PostgreSQL admin.
-- Replace CHANGE_ME_STRONG_PASSWORD before executing.
-- CREATE DATABASE cannot run inside a transaction block in some clients;
-- execute the role section first and the database statement separately.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'apollo_user') THEN
        CREATE ROLE apollo_user
            LOGIN
            PASSWORD 'CHANGE_ME_STRONG_PASSWORD'
            NOSUPERUSER
            NOCREATEDB
            NOCREATEROLE
            NOINHERIT;
    END IF;
END
$$;

-- Run this statement separately if your client reports a transaction error.
CREATE DATABASE eyresqc_apollo OWNER apollo_user;
