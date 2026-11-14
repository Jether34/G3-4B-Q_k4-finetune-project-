param(
    [ValidateSet('extract', 'chat')]
    [string]$Mode = 'extract',
    [double]$Strength = 1.0
)

$ErrorActionPreference = 'Stop'
$script:Root = Split-Path -Parent $PSScriptRoot
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
    Write-Host 'Installing the direct Transformers steering dependencies (no datasets or PyArrow).'
    Invoke-Checked $script:Python @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-Checked $script:Python @('-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu128')
    Invoke-Checked $script:Python @('-m', 'pip', 'install', 'transformers==5.23.0', 'bitsandbytes==0.50.2', 'accelerate==1.15.0', 'safetensors==0.8.0', 'huggingface_hub')
}

$hasToken = & $script:Python -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Sign in to Hugging Face in the browser window opened by the CLI.'
    Invoke-Checked $script:HF @('auth', 'login', '--format', 'agent')
}

Write-Host 'Checking access to google/gemma-3-4b-it...'
try {
    Invoke-Checked $script:Python @('-c', "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='google/gemma-3-4b-it', filename='config.json')")
} catch {
    Write-Host 'The CLI is signed in, but this Hugging Face account cannot download the gated checkpoint yet.'
    Write-Host 'Open https://huggingface.co/google/gemma-3-4b-it, accept/request access, then rerun this script.'
    throw
}

$script:Program = Join-Path $PSScriptRoot 'directional_steering.py'
if ($Mode -eq 'extract') {
    Write-Host 'Extracting local layer directions using Transformers; Heretic and PyArrow are not used.'
    Invoke-Checked $script:Python @($script:Program, 'extract')
} else {
    Write-Host "Starting local chat with runtime steering strength $Strength. Original Ollama weights are unchanged."
    Invoke-Checked $script:Python @($script:Program, 'chat', '--strength', "$Strength")
}
