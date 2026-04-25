@echo off
:: Koba Document Scanner — Stop Hot Folder Watcher
cd /d "%~dp0"
echo.
echo  Stopping Koba Hot Folder Watcher...
powershell -ExecutionPolicy Bypass -File "%~dp0manage.ps1" stop-hotfolder
echo.
pause
