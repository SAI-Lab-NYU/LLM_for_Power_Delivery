$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location (Join-Path $repo "ieee33")
$venvPython = Join-Path $repo ".venv/Scripts/python.exe"
if (Test-Path $venvPython) {
    & $venvPython ieee33_impedance_simulator_web.py @args
} else {
    python ieee33_impedance_simulator_web.py @args
}
