param(
    [int]$K = 5,
    [switch]$SkipFullAnalysis
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

Write-Host "[1/4] Provera Qdrant importa" -ForegroundColor Cyan
& $Python "src\05_verify_import.py"
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "[2/4] Validacija analize gresaka" -ForegroundColor Cyan
& $Python "src\07_error_analysis.py" validate --backend qdrant
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

if (-not $SkipFullAnalysis) {
    Write-Host "[3/4] Analiza svih slika" -ForegroundColor Cyan
    & $Python "src\07_error_analysis.py" analyze --backend qdrant --k $K
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} else {
    Write-Host "[3/4] Puna analiza je preskocena" -ForegroundColor Yellow
}

Write-Host "[4/4] Demo jedne pogresne klasifikacije" -ForegroundColor Cyan
& $Python "src\07_error_analysis.py" demo --backend qdrant --k $K
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

$Report = Join-Path $ProjectRoot "reports\error_analysis\report.html"
if (Test-Path $Report) {
    Write-Host "HTML izvestaj: $Report" -ForegroundColor Green
}

Write-Host "Demo je uspesno zavrsen." -ForegroundColor Green
