<#
.SYNOPSIS
    Build FASL_DualFotography as a standalone executable using PyInstaller.

.DESCRIPTION
    1. Creates/verifies the Python virtual environment
    2. Installs dependencies from requirements.txt
    3. Installs PyInstaller
    4. Runs pyinstaller with build.spec
    5. Output: dist/FASL_DualFotography/FASL_DualFotography.exe

.EXAMPLE
    .\Build_PyInstaller.ps1
#>

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$AppName = "FASL_DualFotography"

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Building $AppName" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

if (-not (Test-Path $VenvPython)) {
    Write-Host "[1/4] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv .venv
} else {
    Write-Host "[1/4] Virtual environment exists" -ForegroundColor Green
}

Write-Host "[2/4] Installing dependencies..." -ForegroundColor Yellow
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -r requirements.txt --quiet
& $VenvPython -m pip install pyinstaller --quiet

Write-Host "[3/4] Running PyInstaller..." -ForegroundColor Yellow
& $VenvPython -m PyInstaller build.spec --clean --noconfirm

$ExePath = Join-Path $ProjectRoot "dist\$AppName\$AppName.exe"
if (Test-Path $ExePath) {
    $Size = [math]::Round((Get-Item $ExePath).Length / 1MB, 1)
    Write-Host "[4/4] SUCCESS: $ExePath ($Size MB)" -ForegroundColor Green
} else {
    Write-Host "[4/4] FAILED: EXE not found at $ExePath" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "To run: dist\$AppName\$AppName.exe" -ForegroundColor Cyan
