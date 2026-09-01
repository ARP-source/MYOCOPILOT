# MyoCopilot — single-container deployment (e.g. Zeabur).
#
# Stage 1 builds the React/Vite frontend. Stage 2 installs Python
# dependencies and serves both the FastAPI backend (WS /stream, /history,
# /stats) and the built frontend from one uvicorn process on one port, via
# serve.py (which mounts frontend/dist/ onto the existing web.server app
# without modifying it — see serve.py's docstring).
#
# Reads OPENROUTER_API_KEY only from the environment (never baked into the
# image) and defaults safely to MOCK=1 whenever no key is supplied — see
# docker-entrypoint.sh.

# ---- Stage 1: frontend build -------------------------------------------
FROM node:20-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ ./
RUN npm run build

# ---- Stage 2: Python runtime --------------------------------------------
FROM python:3.11-slim AS runtime
WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY myosignal/ ./myosignal/
COPY agent/ ./agent/
COPY web/ ./web/
COPY assets/ ./assets/
COPY serve.py ./serve.py
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

COPY --from=frontend-build /app/frontend/dist ./frontend/dist

# Runtime data directories (gesture profiles persist per-container; NinaPro
# / emg2pose files can be mounted at data/ for real-data runs).
RUN mkdir -p /app/data /app/gesture_profiles

EXPOSE 8000
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["./docker-entrypoint.sh"]
