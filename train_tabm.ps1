$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python environment not found: $python"
}
Push-Location -LiteralPath $projectRoot
try {
    & $python -u deep_learning.py `
        --model tabm `
        --numeric-missing-indicators `
        --numeric-embedding ple `
        --numeric-bins 16 `
        --tabm-k 32 `
        --tabm-embedding-dim 16 `
        --hidden-dim 512 `
        --depth 2 `
        --dropout 0.1 `
        --learning-rate 0.002 `
        --weight-decay 0.0003 `
        --folds 5 `
        --epochs 30 `
        --patience 5 `
        --batch-size 512 `
        --output-dir artifacts/tabm_30_epochs_gpu
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
