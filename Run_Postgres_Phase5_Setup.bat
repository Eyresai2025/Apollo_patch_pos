@echo off
setlocal
cd /d "%~dp0"

set DATA_BACKEND=POSTGRESQL
set MONGODB_FALLBACK_ENABLED=False
set MONGODB_MIGRATION_MODE=False

echo ========================================================================
echo Apollo VIT - PostgreSQL Phase 5 Final Runtime Cutover
echo ========================================================================
echo This creates PostgreSQL alarm, repeatability and hardware-test tables.
echo Normal application runtime is validated with MongoDB access disabled.
echo Existing MongoDB data is not deleted.
echo.

python -m pip install -r requirements-postgres.txt
if errorlevel 1 goto :error

python tools\create_postgres_schema.py
if errorlevel 1 goto :error

python tools\check_postgres_connection.py
if errorlevel 1 goto :error

python tools\check_postgres_phase5.py
if errorlevel 1 goto :error

python tools\check_mongodb_disabled_runtime.py
if errorlevel 1 goto :error

python -m unittest tests.test_postgres_phase5 -v
if errorlevel 1 goto :error

echo.
echo [SUCCESS] PostgreSQL Phase 5 completed successfully.
echo Final runtime paths:
echo   Alarm events          -^> PostgreSQL
echo   Repeatability logs    -^> PostgreSQL
echo   Hardware test results -^> PostgreSQL
echo   All Phase 1-4 data    -^> PostgreSQL
echo   MongoDB runtime       -^> DISABLED

echo.
echo Optional remaining-data migration while MongoDB is still available:
echo   Dry run : python tools\migrate_mongodb_operational_data_to_postgres.py
echo   Execute : python tools\migrate_mongodb_operational_data_to_postgres.py --execute

echo.
echo Add these values to the real .env before final GUI testing:
echo   DATA_BACKEND=POSTGRESQL
echo   MONGODB_FALLBACK_ENABLED=False
echo   MONGODB_MIGRATION_MODE=False
pause
exit /b 0

:error
echo.
echo [FAILED] PostgreSQL Phase 5 did not complete. Review the error above.
pause
exit /b 1
