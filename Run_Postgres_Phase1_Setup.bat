@echo off
setlocal
cd /d "%~dp0"

 echo ========================================================================
 echo Apollo VIT - PostgreSQL Phase 1 Setup
 echo ========================================================================
 echo Before continuing, copy .env.postgres.phase1.example values into .env
 echo and replace all CHANGE_ME values.
 echo.

python -m pip install -r requirements-postgres.txt
if errorlevel 1 goto :error

python tools\create_postgres_database.py
if errorlevel 1 goto :error

python tools\create_postgres_schema.py
if errorlevel 1 goto :error

python tools\check_postgres_connection.py
if errorlevel 1 goto :error

python tools\test_postgres_transaction.py
if errorlevel 1 goto :error

python -m unittest tests.test_postgres_phase1 -v
if errorlevel 1 goto :error

echo.
echo [SUCCESS] PostgreSQL Phase 1 completed successfully.
pause
exit /b 0

:error
echo.
echo [FAILED] PostgreSQL Phase 1 did not complete. Review the error above.
pause
exit /b 1
