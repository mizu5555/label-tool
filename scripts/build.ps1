$ErrorActionPreference = "Stop"

$python = if ($env:PYTHON) { $env:PYTHON } else { "python" }
$venv = ".venv-build"
$venvPython = Join-Path $venv "Scripts\python.exe"

function Invoke-Native {
    param(
        [Parameter(ValueFromRemainingArguments = $true)]
        [string[]] $CommandArgs
    )

    & $CommandArgs[0] @($CommandArgs[1..($CommandArgs.Length - 1)])
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $($CommandArgs -join ' ')"
    }
}

Invoke-Native $python -m venv $venv
Invoke-Native $venvPython -m pip install --upgrade pip
Invoke-Native $venvPython -m pip install -r requirements.txt
Invoke-Native $venvPython -m PyInstaller --clean --noconfirm label_tool.spec

Write-Host ""
Write-Host "Build complete: dist\label_tool.exe"
