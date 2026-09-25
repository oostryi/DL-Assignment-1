$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "Project Python environment not found: $python"
}
Push-Location -LiteralPath $projectRoot
try {
    & $python -u deep_learning.py `
        --model transformer `
        --numeric-missing-indicators `
        --numeric-embedding ple `
        --numeric-bins 16 `
        --hidden-dim 64 `
        --depth 2 `
        --batch-size 512 `
        --folds 5 `
        --epochs 30 `
        --patience 5 `
        --output-dir artifacts/transformer_w64_d2_ple_gpu
    if ($LASTEXITCODE -ne 0) {
        throw "Training failed with exit code $LASTEXITCODE"
    }
}
finally {
    Pop-Location
}
