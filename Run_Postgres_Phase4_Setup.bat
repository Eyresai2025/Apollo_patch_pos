@echo off
setlocal
cd /d "%~dp0"

echo ========================================================================
echo Apollo VIT - PostgreSQL Phase 4A Setup
echo ========================================================================
echo This creates chunked PostgreSQL binary storage for inspection and New SKU images.
echo Existing MongoDB GridFS data is not deleted and remains available as fallback.
echo.

python -m pip install -r requirements-postgres.txt
if errorlevel 1 goto :error

python tools\create_postgres_schema.py
if errorlevel 1 goto :error

python tools\check_postgres_connection.py
if errorlevel 1 goto :error

python tools\check_postgres_phase4.py
if errorlevel 1 goto :error

python -m unittest tests.test_postgres_phase4 -v
if errorlevel 1 goto :error

echo.
echo [SUCCESS] PostgreSQL Phase 4A completed successfully.
echo New image data paths:
echo   Binary metadata       -^> apollo.file_assets
echo   Binary chunks         -^> apollo.file_asset_chunks
echo   Inspection image map  -^> apollo.inspection_images
echo   New SKU image map     -^> apollo.new_sku_images
echo   Legacy MongoDB GridFS -^> read-only fallback until final cutover
echo.
echo Optional legacy image migrations:
echo   Inspection dry run : python tools\migrate_gridfs_images_to_postgres.py
echo   Inspection execute : python tools\migrate_gridfs_images_to_postgres.py --execute
echo   New SKU dry run    : python tools\migrate_mongodb_new_sku_images_to_postgres.py
echo   New SKU execute    : python tools\migrate_mongodb_new_sku_images_to_postgres.py --execute
pause
exit /b 0

:error
echo.
echo [FAILED] PostgreSQL Phase 4A did not complete. Review the error above.
pause
exit /b 1
