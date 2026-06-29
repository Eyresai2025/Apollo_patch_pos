@echo off
setlocal
cd /d "%~dp0"

echo ========================================================================
echo Apollo VIT - PostgreSQL Phase 3 Setup
echo ========================================================================
echo This applies inspection-cycle metadata and event tables.
echo Inspection image binaries remain in the existing MongoDB GridFS buckets.
echo Existing MongoDB data is not deleted.
echo.

python -m pip install -r requirements-postgres.txt
if errorlevel 1 goto :error

python tools\create_postgres_schema.py
if errorlevel 1 goto :error

python tools\check_postgres_connection.py
if errorlevel 1 goto :error

python tools\check_postgres_phase3.py
if errorlevel 1 goto :error

python -m unittest tests.test_postgres_phase3 -v
if errorlevel 1 goto :error

echo.
echo [SUCCESS] PostgreSQL Phase 3 completed successfully.
echo New application data paths:
echo   Inspection metadata -^> apollo.inspection_cycles
echo   Inspection events   -^> apollo.inspection_cycle_events
echo   Input/output images -^> existing MongoDB GridFS until Phase 4
echo.
echo Optional existing-data migration:
echo   Dry run : python tools\migrate_mongodb_inspections_to_postgres.py
echo   Execute : python tools\migrate_mongodb_inspections_to_postgres.py --execute
pause
exit /b 0

:error
echo.
echo [FAILED] PostgreSQL Phase 3 did not complete. Review the error above.
pause
exit /b 1
