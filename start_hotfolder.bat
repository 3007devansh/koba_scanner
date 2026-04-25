@echo off
:: Koba Document Scanner — Start Hot Folder Watcher
:: Double-click to start watching the hot\input folder.
:: Edit --folder or --interval below if needed.
cd /d "%~dp0"
echo.
echo  Starting Koba Hot Folder Watcher...
powershell -ExecutionPolicy Bypass -File "%~dp0manage.ps1" start-hotfolder -Folder "hot" -Interval 5
echo.
pause
