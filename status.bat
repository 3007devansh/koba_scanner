@echo off
:: Koba Document Scanner — Service Status
:: Shows whether the web app and hot folder are running, plus recent log lines.
cd /d "%~dp0"
powershell -ExecutionPolicy Bypass -File "%~dp0manage.ps1" status
pause
