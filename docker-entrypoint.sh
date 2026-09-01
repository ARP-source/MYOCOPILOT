#!/bin/sh
# MyoCopilot container entrypoint.
#
# Safe-by-default: if neither OPENROUTER_API_KEY nor MOCK was supplied, force
# MOCK=1 so the container never silently tries (and fails) to call OpenRouter
# with no key. An explicit MOCK=0/MOCK=1 from the caller is always honored.
set -e

if [ -z "${OPENROUTER_API_KEY:-}" ] && [ -z "${MOCK:-}" ]; then
    export MOCK=1
fi

PORT="${PORT:-8000}"
exec uvicorn serve:app --host 0.0.0.0 --port "$PORT"
