# MyoCopilot test suite — Python tests + frontend build.
$ErrorActionPreference = "Stop"

py -m pytest
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Push-Location frontend
try {
    npm run build
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
} finally {
    Pop-Location
}
