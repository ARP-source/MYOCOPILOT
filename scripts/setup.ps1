# MyoCopilot setup, installs Python and frontend dependencies.
$ErrorActionPreference = "Stop"

py -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Push-Location frontend
try {
    npm install
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
