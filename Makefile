# MyoCopilot — demo orchestration.
#
# GNU Make on Windows (Git Bash / WSL / choco make) can run this directly.
# If `make` isn't available, use the PowerShell equivalents in scripts/ —
# e.g. `.\scripts\demo.ps1` instead of `make demo`. See README.md.

.PHONY: setup test backend frontend demo eval report

setup:
	py -m pip install -r requirements.txt
	cd frontend && npm install

test:
	py -m pytest
	cd frontend && npm run build

# MOCK=1: never calls OpenRouter, never needs OPENROUTER_API_KEY.
backend:
	MOCK=1 py -m uvicorn web.server:app --reload --port 8000

frontend:
	cd frontend && npm run dev

# Headless, deterministic, MOCK=1 — no browser, no API key. See eval/demo.py.
demo:
	MOCK=1 py -m eval.demo

# Offline evaluation harness + classical baseline. MOCK-only, no API key.
eval:
	py -m eval.run
	py -m eval.baselines

# Regenerates eval/report/*.png from eval/results.json.
report:
	py -m eval.report
