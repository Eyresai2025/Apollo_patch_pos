@echo off
setlocal
cd /d "%~dp0"

echo ========================================================================
echo Apollo VIT - PostgreSQL Phase 2 Setup
echo ========================================================================
echo This applies SKU, recipe, active-recipe and device-profile tables.
echo Existing MongoDB files and collections are not deleted.
echo.

python -m pip install -r requirements-postgres.txt
if errorlevel 1 goto :error

python tools\create_postgres_schema.py
if errorlevel 1 goto :error

python tools\check_postgres_connection.py
if errorlevel 1 goto :error

python tools\check_postgres_phase2.py
if errorlevel 1 goto :error

python -m unittest tests.test_postgres_phase2 -v
if errorlevel 1 goto :error

echo.
echo [SUCCESS] PostgreSQL Phase 2 completed successfully.
echo New application data paths:
echo   New SKU setup  -^> apollo.skus
echo   SKU recipes    -^> apollo.sku_recipes
ECHO   Active state   -^> apollo.active_recipe_state
echo   Device profiles-^> apollo.device_profiles
pause
exit /b 0

:error
echo.
echo [FAILED] PostgreSQL Phase 2 did not complete. Review the error above.
pause
exit /b 1
