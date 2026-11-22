"""
serve, production ASGI entrypoint for Docker / Zeabur.

Imports the existing FastAPI app from web.server UNCHANGED and mounts the
built React frontend (frontend/dist/) onto it as a catch-all static route,
so one process on one port serves both the API/WebSocket and the dashboard.
web/server.py itself is never edited, the mount is attached here, from
outside, onto the same `app` object, after its own routes (/stream,
/history, /stats) are already registered, so those still take priority and
everything else falls through to the SPA's static files.

The frontend already talks same-origin (frontend/src/store/useStore.ts
builds its WebSocket URL from window.location.host and fetches "/history"
and "/stats" as relative paths), so no frontend changes are needed either.

Usage:
    uvicorn serve:app --host 0.0.0.0 --port 8000
"""

from pathlib import Path

from fastapi.staticfiles import StaticFiles

from web.server import app

_FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

if _FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIST), html=True), name="frontend")
