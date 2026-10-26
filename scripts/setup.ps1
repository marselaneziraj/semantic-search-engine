# Creates the virtual environment and installs everything.
# Run from the repository root:   .\scripts\setup.ps1
# PowerShell will not run a script without the leading .\

$ErrorActionPreference = "Stop"

# torch (pulled in by sentence-transformers) has no wheels for Python 3.14 yet,
# so prefer 3.12, then 3.11, then whatever `python` resolves to.
$py = $null
foreach ($candidate in @("3.12", "3.11", "3.13")) {
    if (py -$candidate -c "print(1)" 2>$null) { $py = "py -$candidate"; break }
}
if (-not $py) { $py = "python" }
Write-Host "Using interpreter: $py" -ForegroundColor Cyan

if (-not (Test-Path .\.venv)) {
    Invoke-Expression "$py -m venv .venv"
}
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -e ".[dense,eval,serve,dev]"

Write-Host ""
Write-Host "Done. Next:" -ForegroundColor Green
Write-Host "  .\scripts\smoke.ps1        # offline check, no downloads (about 30 seconds)"
Write-Host "  .\scripts\benchmark.ps1    # the real SciFact run (downloads models + data)"
