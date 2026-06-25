# Apollo VIT — PostgreSQL Phase 1 Setup

## Purpose

Phase 1 creates and validates a PostgreSQL foundation **without changing the
current MongoDB application flow**. MongoDB remains safe and operational while
PostgreSQL is tested in parallel.

## Files added

```text
.env.postgres.phase1.example
requirements-postgres.txt
Run_Postgres_Phase1_Setup.bat
PHASE1_POSTGRES_SETUP_README.md

database/
  migrations/001_phase1_foundation.sql
  setup/00_create_database.sql

src/COMMON/postgres/
  __init__.py
  settings.py
  connection.py
  migrations.py
  health.py

tools/
  create_postgres_database.py
  create_postgres_schema.py
  check_postgres_connection.py
  test_postgres_transaction.py

tests/
  test_postgres_phase1.py
```

## Files intentionally not modified in Phase 1

```text
src/COMMON/db.py
src/COMMON/recipe_service.py
src/COMMON/inspection_repository.py
src/COMMON/inspection_image_store.py
GUI.py
```

These still use MongoDB. PostgreSQL business repositories will replace them
in later phases only after the connection and schema are proven stable.

## Step 1 — Install PostgreSQL

Install PostgreSQL on the Edge IPC and remember the administrator password for
the `postgres` user. Confirm the PostgreSQL Windows service is running.

## Step 2 — Add PostgreSQL values to `.env`

Open `.env.postgres.phase1.example`. Copy its complete block to the bottom of
the existing project `.env`.

Replace:

```text
CHANGE_ME_ADMIN_PASSWORD
CHANGE_ME_STRONG_PASSWORD
```

Use the same application password in `POSTGRES_APP_PASSWORD` and
`POSTGRES_DATABASE_URL`.

Do not remove the existing MongoDB values during Phase 1.

## Step 3 — Activate the Apollo environment

Example:

```bat
conda activate Apollo
cd /d C:\Users\Hi\OneDrive - radometech.com\Desktop\Apollo_Vit_App_postgres
```

## Step 4 — Run the automatic setup

```bat
Run_Postgres_Phase1_Setup.bat
```

The script performs:

1. Installs Psycopg and the connection-pool package.
2. Creates `apollo_user` if missing.
3. Creates `eyresqc_apollo` if missing.
4. Creates the `apollo` schema.
5. Applies numbered SQL migrations.
6. Checks the connection.
7. Validates commit and rollback.
8. Runs the Phase 1 integration tests.

## Manual commands

Run these when troubleshooting:

```bat
python -m pip install -r requirements-postgres.txt
python tools\create_postgres_database.py
python tools\create_postgres_schema.py
python tools\check_postgres_connection.py
python tools\test_postgres_transaction.py
python -m unittest tests.test_postgres_phase1 -v
```

## Expected successful output

```text
[OK] Application role created or already exists.
[OK] Application database created or already exists.
[APPLIED] 001_phase1_foundation.sql
[OK] Connected
[OK] Commit test passed.
[OK] Rollback test passed.
[SUCCESS] PostgreSQL Phase 1 completed successfully.
```

## Database objects created

```text
eyresqc_apollo
└── schema: apollo
    ├── schema_migrations
    ├── application_settings
    └── database_events
```

## Important safety rule

Do not set the main application database implementation to PostgreSQL yet.
Phase 1 only proves PostgreSQL connectivity, pooling, migrations, transactions,
and schema ownership. Continue running the normal GUI with MongoDB until Phase
2 replaces New SKU, Recipe, Active Recipe and Device Profile persistence.
