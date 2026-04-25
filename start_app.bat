@echo off
:: Koba Document Scanner — Start Web Dashboard
:: Double-click this file to launch the web app in the background.
cd /d "%~dp0"
echo.
echo  Starting Koba Document Scanner...
powershell -ExecutionPolicy Bypass -File "%~dp0manage.ps1" start-app
echo.
pause
