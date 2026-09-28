$ErrorActionPreference = "Stop"

$ProjectRoot = $PSScriptRoot
$Venv = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"
$Requirements = Join-Path $ProjectRoot "code\requirements.txt"

if (-not (Test-Path $Python)) {
    py -3.11 -m venv $Venv
    if ($LASTEXITCODE -ne 0) { throw "Could not create .venv. Install Python 3.11 and rerun." }
}

& $Python -m pip install -r $Requirements
if ($LASTEXITCODE -ne 0) { throw "Dependency installation failed." }

$Check = "import numpy, torch, ortools, sklearn, matplotlib, optuna; print('PyTorch:', torch.__version__); print('Optuna:', optuna.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
& $Python -c $Check
if ($LASTEXITCODE -ne 0) { throw "Environment check failed." }

Write-Host "Environment ready. Run the smoke test with: cd code; ..\.venv\Scripts\python.exe train.py --smoke-test"
