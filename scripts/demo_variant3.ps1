param(
    [double]$Threshold = 0.94
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

if (Test-Path $VenvPython) {
    $Python = $VenvPython
} else {
    $Python = "python"
}

Set-Location $ProjectRoot

Write-Host "[1/6] Provera Qdrant importa" -ForegroundColor Cyan
& $Python "src\05_verify_import.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/6] Validacija varijante 3" -ForegroundColor Cyan
& $Python "src\08_dataset_cleaning.py" validate --backend qdrant
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[3/6] Analiza slicnih slika" -ForegroundColor Cyan
& $Python "src\08_dataset_cleaning.py" analyze --backend qdrant --threshold $Threshold
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[4/6] Prikaz prve grupe" -ForegroundColor Cyan
& $Python "src\08_dataset_cleaning.py" inspect-group 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[5/6] Pravljenje ociscene kopije" -ForegroundColor Cyan
& $Python "src\08_dataset_cleaning.py" build-clean-dataset
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[6/6] Provera ociscene kopije" -ForegroundColor Cyan
& $Python "src\08_dataset_cleaning.py" verify-cleaned
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Report = Join-Path $ProjectRoot "reports\variant3_dataset_cleaning\report.html"
Write-Host "Varijanta 3 je uspesno zavrsena." -ForegroundColor Green
Write-Host "HTML izvestaj: $Report" -ForegroundColor Green
