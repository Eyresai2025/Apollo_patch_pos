@echo off
setlocal

cd /d C:\Apollo_patch_pos

if not exist backups\postgres mkdir backups\postgres

for /f "tokens=1-4 delims=/ " %%a in ("%date%") do (
    set DD=%%a
    set MM=%%b
    set YYYY=%%c
)

for /f "tokens=1-2 delims=: " %%a in ("%time%") do (
    set HH=%%a
    set MIN=%%b
)

set BACKUP_FILE=backups\postgres\eyresqc_apollo_%YYYY%-%MM%-%DD%_%HH%%MIN%.backup

echo Creating PostgreSQL backup...
echo Output: %BACKUP_FILE%

pg_dump -h 127.0.0.1 -U apollo_user -d eyresqc_apollo -F c -f "%BACKUP_FILE%"

if %ERRORLEVEL% neq 0 (
    echo Backup failed.
    pause
    exit /b 1
)

echo Backup completed successfully.
pause