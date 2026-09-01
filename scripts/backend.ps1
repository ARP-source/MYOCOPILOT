# MyoCopilot backend — MOCK=1: never calls OpenRouter, never needs OPENROUTER_API_KEY.
$env:MOCK = "1"
py -m uvicorn web.server:app --reload --port 8000
