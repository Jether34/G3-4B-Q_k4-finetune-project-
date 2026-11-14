$ErrorActionPreference = 'Stop'

$script:Root = Split-Path -Parent $PSScriptRoot
$script:Venv = Join-Path $PSScriptRoot '.venv'
$script:Python = Join-Path $script:Venv 'Scripts\python.exe'
$script:HF = Join-Path $script:Venv 'Scripts\hf.exe'
$script:ModelId = 'google/gemma-3-4b-it'

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

if (-not (Test-Path -LiteralPath $script:HF)) {
    Write-Host 'Installing the small Hugging Face client into the isolated environment...'
    Invoke-Checked $script:Python @('-m', 'pip', 'install', '--upgrade', 'pip')
    Invoke-Checked $script:Python @('-m', 'pip', 'install', 'huggingface_hub')
}

$env:HERETIC_CONFIG_FILE = Join-Path $PSScriptRoot 'config.toml'

# Authenticate and check gated-repository access before downloading CUDA wheels
# or model weights. Credentials stay in the normal Hugging Face user cache.
$hasToken = & $script:Python -c "from huggingface_hub import get_token; import sys; sys.exit(0 if get_token() else 1)"
if ($LASTEXITCODE -ne 0) {
    Write-Host 'Sign in to Hugging Face. The Gemma model page requires access approval first:'
    Write-Host 'https://huggingface.co/google/gemma-3-4b-it'
    Invoke-Checked $script:HF @('auth', 'login', '--format', 'agent')
}

Write-Host "Checking access to $script:ModelId ..."
try {
    Invoke-Checked $script:Python @('-c', "from huggingface_hub import HfApi; HfApi().model_info('$script:ModelId')")
} catch {
    Write-Host 'The Hugging Face account must have access to the Gemma checkpoint. Accept its terms on the model page, then rerun this script.'
    throw
}

if (-not (Test-Path -LiteralPath (Join-Path $script:Venv 'Scripts\heretic.exe'))) {
    Write-Host 'Installing PyTorch CUDA 12.8 and Heretic in the isolated environment...'
    Invoke-Checked $script:Python @('-m', 'pip', 'install', 'torch', 'torchvision', '--index-url', 'https://download.pytorch.org/whl/cu128')
    Invoke-Checked $script:Python @('-m', 'pip', 'install', 'heretic-llm==2.1.0')
}

Write-Host 'Starting the configured local run. The original Ollama model is unchanged.'
$env:USE_HUB_KERNELS = 'NO'
$kernelsInstalled = & $script:Python -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('kernels') else 1)"
if ($LASTEXITCODE -eq 0) {
    Write-Host 'Removing the optional Transformers Hub-kernels package from this isolated environment.'
    Invoke-Checked $script:Python @('-m', 'pip', 'uninstall', '-y', 'kernels')
}

Push-Location $PSScriptRoot
try {
    Invoke-Checked (Join-Path $script:Venv 'Scripts\heretic.exe') @()
} finally {
    Pop-Location
}
