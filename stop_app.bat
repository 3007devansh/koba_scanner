@echo off
:: Koba Document Scanner — Stop Web Dashboard
cd /d "%~dp0"
echo.
echo  Stopping Koba Document Scanner...
powershell -ExecutionPolicy Bypass -File "%~dp0manage.ps1" stop-app
echo.
pause
