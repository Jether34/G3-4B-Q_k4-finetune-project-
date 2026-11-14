param(
    [string]$OutputDir = (Join-Path (Split-Path -Parent $PSScriptRoot) 'artifacts\gemma-3-4b-it-abliterated'),
    [string]$Prompts = (Join-Path $PSScriptRoot 'contrastive_prompts.json')
)

$ErrorActionPreference = 'Stop'
$script:Venv = Join-Path $PSScriptRoot '.venv'
$script:Python = Join-Path $script:Venv 'Scripts\python.exe'
$script:HF = Join-Path $script:Venv 'Scripts\hf.exe'

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed ($LASTEXITCODE): $Executable $($Arguments -join ' ')"
    }
}

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw 'Install Python 3.12 (py launcher) and rerun this script.'
}

if (-not (Test-Path -LiteralPath $script:Python)) {
    Invoke-Checked 'py' @('-3.12', '-m', 'venv', $script:Venv)
}

$env:USE_HUB_KERNELS = 'NO'
$env:HF_HUB_DISABLE_TELEMETRY = '1'
$ready = & $script:Python -c "import torch, transformers, bitsandbytes, accelerate, safetensors, huggingface_hub"
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Installing direct Transformers weight-edit dependencies.'
    Invoke-Checked $script:Python @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-Checked $script:Python @('-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu128')
    Invoke-Checked $script:Python @('-m', 'pip', 'install', 'transformers==5.23.0', 'bitsandbytes==0.50.2', 'accelerate==1.15.0', 'safetensors==0.8.0', 'huggingface_hub')
}

$hasToken = & $script:Python -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Sign in to Hugging Face in the browser window opened by the CLI.'
    Invoke-Checked $script:HF @('auth', 'login', '--format', 'agent')
}

Write-Host 'Preparing a separate BF16 weight-edited checkpoint. The cached source is never overwritten.'
$script:Program = Join-Path $PSScriptRoot 'abliterate_weights.py'
Invoke-Checked $script:Python @($script:Program, '--prompts', $Prompts, '--output', $OutputDir)
