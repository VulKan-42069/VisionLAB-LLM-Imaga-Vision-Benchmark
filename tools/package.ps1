# Build a release VisionLab.exe and compile VisionLab-Setup-<version>.exe
# Requires Inno Setup 6 (ISCC.exe).

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$Iss = Join-Path $PSScriptRoot "installer\visionlab.iss"
$IsccCandidates = @(
    "$env:LOCALAPPDATA\Programs\Inno Setup 6\ISCC.exe",
    "${env:ProgramFiles(x86)}\Inno Setup 6\ISCC.exe",
    "$env:ProgramFiles\Inno Setup 6\ISCC.exe"
)
$Iscc = $IsccCandidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $Iscc) {
    throw "Inno Setup 6 was not found. Install it, then re-run tools/package.ps1."
}

Get-Process -Name visionlab -ErrorAction SilentlyContinue | Stop-Process -Force -ErrorAction SilentlyContinue
Start-Sleep -Milliseconds 400

$env:SLINT_BACKEND = "winit-skia"
Set-Location $Root
Write-Host "Building release visionlab.exe..."
& "$env:USERPROFILE\.cargo\bin\cargo.exe" build --release
if ($LASTEXITCODE -ne 0) {
    throw "cargo build --release failed."
}

$Exe = Join-Path $Root "target\release\visionlab.exe"
if (-not (Test-Path $Exe)) {
    throw "Missing $Exe"
}

New-Item -ItemType Directory -Force -Path (Join-Path $Root "dist") | Out-Null
Write-Host "Compiling installer with $Iscc..."
& $Iscc $Iss
if ($LASTEXITCODE -ne 0) {
    throw "ISCC failed."
}

Get-ChildItem (Join-Path $Root "dist\VisionLab-Setup-*.exe") | ForEach-Object {
    Write-Host ("Installer: {0} ({1:N1} MB)" -f $_.FullName, ($_.Length / 1MB))
}
