@echo off
setlocal EnableExtensions DisableDelayedExpansion

cd /d "%~dp0.."

where git >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Git is not available in PATH.
    exit /b 2
)

git rev-parse --is-inside-work-tree >nul 2>nul
if errorlevel 1 (
    echo [ERROR] This folder is not a Git repository.
    exit /b 2
)

echo ================================================================================
echo AP-002 Apollo Git Runtime Cleanup
echo ================================================================================
echo.
echo The following files are already tracked by Git but should now be ignored:
echo.
git ls-files -ci --exclude-standard

echo.
echo IMPORTANT:
echo   This cleanup uses "git rm --cached".
echo   It REMOVES files from Git tracking only.
echo   It DOES NOT delete the actual local files from this Apollo PC.
echo.
choice /C YN /N /M "Continue and untrack the files shown above? [Y/N]: "
if errorlevel 2 (
    echo Cancelled. No Git index changes were made.
    exit /b 0
)

echo.
echo Untracking ignored runtime files...
for /f "usebackq delims=" %%F in (`git ls-files -ci --exclude-standard`) do (
    git rm --cached --ignore-unmatch -- "%%F"
    if errorlevel 1 (
        echo [ERROR] Failed while untracking: %%F
        exit /b 3
    )
)

echo.
echo ================================================================================
echo Cleanup complete.
echo Local files were preserved. Review Git status before committing.
echo ================================================================================
git status --short

echo.
echo Recommended verification:
echo   python tools\audit_repository_hygiene.py
echo   python GUI.py

echo.
endlocal
