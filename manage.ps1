# =============================================================================
#  Koba Document Scanner - manage.ps1
#  Windows background-process management script.
# =============================================================================

param(
    [Parameter(Position=0)]
    [ValidateSet("start-app","stop-app","start-hotfolder","stop-hotfolder","status","restart-app")]
    [string]$Command,

    [string]$Folder   = "hot",
    [int]   $Interval = 5,
    [int]   $Port     = 5000
)

$ErrorActionPreference = "Continue" # Don't crash on minor warnings
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
if (-not $ScriptDir) { $ScriptDir = Get-Location }
Set-Location $ScriptDir

# -- helper: locate Python -----------------------------------------------------
function Get-Python {
    $venvPath = Join-Path $ScriptDir ".venv\Scripts"
    $pythonw  = Join-Path $venvPath "pythonw.exe"
    $python   = Join-Path $venvPath "python.exe"

    if (Test-Path $pythonw) { return $pythonw }
    if (Test-Path $python)  { return $python }
    
    # Try system python
    $sysPython = where.exe python.exe 2>$null | Select-Object -First 1
    if ($sysPython) { return $sysPython }

    return $null
}

# -- helper: check if process is actually running -----------------------------
function Test-Pid {
    param([string]$PidFile)
    if (-not (Test-Path $PidFile)) { return $false }
    
    $content = Get-Content $PidFile -ErrorAction SilentlyContinue
    if (-not $content) { return $false }
    $id = $content.ToString().Trim()
    if (-not $id) { return $false }

    try {
        $proc = Get-Process -Id ([int]$id) -ErrorAction SilentlyContinue
        return ($null -ne $proc)
    } catch {
        return $false
    }
}

# -- helper: stop process ------------------------------------------------------
function Stop-ByPidFile {
    param([string]$PidFile, [string]$Name)
    if (-not (Test-Path $PidFile)) {
        Write-Host "  [$Name] No PID entry found. Is it running?"
        return
    }
    
    $id = (Get-Content $PidFile).Trim()
    try {
        $proc = Get-Process -Id ([int]$id) -ErrorAction SilentlyContinue
        if ($proc) {
            Stop-Process -Id ([int]$id) -Force
            Write-Host "  [$Name] Stopped (PID $id)"
        } else {
            Write-Host "  [$Name] Process $id is already closed."
        }
    } catch {
        Write-Host "  [$Name] Could not stop process $id (maybe already closed)."
    }
    Remove-Item $PidFile -ErrorAction SilentlyContinue
}

# -- helper: status banner -----------------------------------------------------
function Write-Banner {
    param([string]$Title, [string[]]$Lines)
    $w = 50
    Write-Host "+" + ("-" * ($w-2)) + "+"
    Write-Host "|  $Title".PadRight($w-1) + "|"
    Write-Host "+" + ("-" * ($w-2)) + "+"
    foreach ($l in $Lines) {
        Write-Host "|  $l".PadRight($w-1) + "|"
    }
    Write-Host "+" + ("-" * ($w-2)) + "+"
}

# =============================================================================
$pythonPath = Get-Python

switch ($Command) {

    "start-app" {
        if (-not $pythonPath) { throw "Could not find Python! Please ensure .venv is installed." }
        if (Test-Pid "app.pid") {
            Write-Host "Koba Web App is already running."
            break
        }

        $proc = Start-Process -FilePath $pythonPath -ArgumentList "main.py", "app" `
            -WorkingDirectory $ScriptDir -WindowStyle Hidden -PassThru
        
        Start-Sleep -Seconds 1
        $proc.Id | Out-File "app.pid" -Encoding ascii
        
        Write-Banner "Koba Web App Started" @(
            "URL: http://localhost:$Port",
            "PID: $($proc.Id)",
            "Log: app.log"
        )
    }

    "stop-app" {
        Stop-ByPidFile "app.pid" "Koba Web App"
    }

    "start-hotfolder" {
        if (-not $pythonPath) { throw "Could not find Python! Please ensure .venv is installed." }
        if (Test-Pid "hotfolder.pid") {
            Write-Host "Koba Hot Folder is already running."
            break
        }

        $absFolder = (Resolve-Path $Folder -ErrorAction SilentlyContinue).Path
        if (-not $absFolder) { $absFolder = Join-Path $ScriptDir $Folder }

        $proc = Start-Process -FilePath $pythonPath -ArgumentList "hot_folder.py", "--folder", "`"$absFolder`"", "--interval", $Interval `
            -WorkingDirectory $ScriptDir -WindowStyle Hidden -PassThru
        
        Start-Sleep -Seconds 1
        $proc.Id | Out-File "hotfolder.pid" -Encoding ascii
        
        Write-Banner "Koba Hot Folder Started" @(
            "Folder: $absFolder",
            "PID   : $($proc.Id)",
            "Log   : hot_folder.log"
        )
    }

    "stop-hotfolder" {
        Stop-ByPidFile "hotfolder.pid" "Koba Hot Folder"
    }

    "status" {
        Write-Host "`n--- Koba Scanner Service Status ---"
        
        $appRun = Test-Pid "app.pid"
        $hfRun  = Test-Pid "hotfolder.pid"

        if ($appRun) { 
            $foundPid = (Get-Content "app.pid").Trim()
            Write-Host " [Web App]    : RUNNING (PID $foundPid) - http://localhost:$Port" -ForegroundColor Green
        } else { 
            Write-Host " [Web App]    : STOPPED" -ForegroundColor Gray
        }

        if ($hfRun) { 
            $foundPid = (Get-Content "hotfolder.pid").Trim()
            Write-Host " [Hot Folder] : RUNNING (PID $foundPid)" -ForegroundColor Green
        } else { 
            Write-Host " [Hot Folder] : STOPPED" -ForegroundColor Gray
        }
        
        Write-Host "------------------------------------`n"
        
        if ($appRun -and (Test-Path "app.log")) {
            Write-Host "Recent app activity:"
            Get-Content "app.log" | Select-Object -Last 3
        }
        if ($hfRun -and (Test-Path "hot_folder.log")) {
            Write-Host "Recent hot folder activity:"
            Get-Content "hot_folder.log" | Select-Object -Last 3
        }
    }

    default {
        Write-Host "Usage: .\manage.ps1 [status | start-app | stop-app | start-hotfolder | stop-hotfolder]"
    }
}
