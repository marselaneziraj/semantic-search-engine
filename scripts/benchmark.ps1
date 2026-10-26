# The real benchmark run: downloads all-MiniLM-L6-v2 (~90 MB), the cross-encoder
# (~80 MB) and the BEIR dataset via ir_datasets, then evaluates all four systems.
# Run from the repository root:   .\scripts\benchmark.ps1
# Optional first argument picks the dataset:  .\scripts\benchmark.ps1 beir/nfcorpus

param([string]$Dataset = "beir/scifact")

$ErrorActionPreference = "Stop"
if (Test-Path .\.venv\Scripts\Activate.ps1) { .\.venv\Scripts\Activate.ps1 }

Write-Host "Evaluating $Dataset - first run embeds the corpus and is the slow one." -ForegroundColor Cyan
semsearch eval --dataset $Dataset

Write-Host "Ablations (chunk size, rerank depth)" -ForegroundColor Cyan
semsearch ablate --dataset $Dataset --kind chunk
semsearch ablate --dataset $Dataset --kind topk

Write-Host "Results are in .\results" -ForegroundColor Green
