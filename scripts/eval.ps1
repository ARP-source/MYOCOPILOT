# MyoCopilot offline evaluation harness + classical baseline. MOCK-only, no API key.
$ErrorActionPreference = "Stop"

py -m eval.run
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

py -m eval.baselines
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
