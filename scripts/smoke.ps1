# Offline end-to-end check: no model downloads, no dataset downloads.
# Run from the repository root:   .\scripts\smoke.ps1

$ErrorActionPreference = "Stop"
if (Test-Path .\.venv\Scripts\Activate.ps1) { .\.venv\Scripts\Activate.ps1 }

Write-Host "== tests ==" -ForegroundColor Cyan
pytest -q

Write-Host "== offline evaluation (toy corpus, hashed embeddings) ==" -ForegroundColor Cyan
semsearch eval --config configs/experiment.offline.yaml

Write-Host "== sample queries ==" -ForegroundColor Cyan
semsearch search --config configs/experiment.offline.yaml --system hybrid `
    -q "how do you combine two ranked lists" `
    -q "why does keyword search fail when wording differs"
