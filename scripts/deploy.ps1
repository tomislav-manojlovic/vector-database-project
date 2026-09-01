$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$ComposeFile = Join-Path $ProjectRoot "infra\docker-compose.yml"

Set-Location $ProjectRoot

if (-not (Test-Path $Python)) {
    Write-Error "Nije pronadjen .venv. Prvo napravi virtuelno okruzenje i instaliraj requirements.txt."
    exit 1
}

if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "Docker nije pronadjen. Instaliraj ili pokreni Docker Desktop."
    exit 1
}

Write-Host "[1/7] Pokretanje Qdranta" -ForegroundColor Cyan
& docker compose -f $ComposeFile up -d
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/7] Cekanje da Qdrant bude spreman" -ForegroundColor Cyan
& $Python "infra\scripts\wait_for_qdrant.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/7] Provera embedding fajlova" -ForegroundColor Cyan
& $Python "src\check_embeddings.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/7] Kreiranje Qdrant kolekcije" -ForegroundColor Cyan
& $Python "src\02_create_collection.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[5/7] Import podataka u Qdrant" -ForegroundColor Cyan
& $Python "src\04_import_to_qdrant.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[6/7] Verifikacija importa" -ForegroundColor Cyan
& $Python "src\05_verify_import.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[7/7] Pokretanje korisnickog interfejsa" -ForegroundColor Cyan
Write-Host "Deploy je uspesno zavrsen." -ForegroundColor Green
& $Python "ui\server.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
