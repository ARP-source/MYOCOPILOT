# MyoCopilot — CLAUDE.md

## Architecture Overview

MyoCopilot is a conversational myoelectric control copilot.  Public sEMG datasets
(NinaPro DB2 .mat files, Meta emg2pose mini HDF5) are replayed as a simulated live
stream, converted to spectrogram images, and classified by GLM-5.3-Flash via
OpenRouter (model id: `z-ai/glm-5.3-flash`).

The agent detects accuracy drift (simulated electrode shift / sweat / channel swap)
and recalibrates conversationally: it interviews the user, requests fresh samples of
the confused gesture, and updates a per-user gesture profile with few-shot spectrogram
examples.  Recognized gestures drive the 3-D hand at `assets/hand.html`.  A dashboard
shows live rolling accuracy, a confusion matrix, and API cost.

```
┌─────────────────────────────────────────────────────────────┐
│                         myosignal/                          │
│  loader.py   → reads NinaPro / emg2pose files               │
│  windower.py → sliding-window + feature extraction          │
│  spectro.py  → STFT → PNG → base64                          │
│  drift.py    → injects electrode_shift / sweat / channel_swap│
│  stream.py   → Iterator[WindowEnvelope]  ← PUBLIC API       │
└────────────────────┬────────────────────────────────────────┘
                     │ WindowEnvelope (JSON)
┌────────────────────▼────────────────────────────────────────┐
│                          agent/                             │
│  classifier.py → calls OpenRouter GLM-5.3-Flash             │
│  drift_detector.py → rolling accuracy + drift heuristics    │
│  recalibrator.py   → conversational interview + profile update│
│  profile_store.py  → load/save gesture_profiles/*.json      │
│  cost_tracker.py   → token counting + USD accounting        │
└────────────────────┬────────────────────────────────────────┘
                     │ ClassifierResponse + events
┌────────────────────▼────────────────────────────────────────┐
│                           web/                              │
│  server.py  → FastAPI app (WS /stream, HTTP endpoints)      │
│  session.py → per-connection state (history, stats)         │
└────────────────────┬────────────────────────────────────────┘
                     │ WS messages
┌────────────────────▼────────────────────────────────────────┐
│                          assets/                            │
│  hand.html      → 3-D hand driven by gesture labels         │
│  dashboard.html → live accuracy, confusion matrix, cost     │
└─────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                           eval/                             │
│  bench.py    → offline accuracy / latency benchmarks        │
│  report.py   → generate confusion matrix plots              │
└──────────────────────────────────────────────────────────────┘
```

---

## Module Ownership Map

| Folder           | Owner session | Allowed to edit                          |
|------------------|---------------|------------------------------------------|
| `myosignal/`     | signal-session| `myosignal/**`                           |
| `agent/`         | agent-session | `agent/**`, `gesture_profiles/**`        |
| `web/`           | web-session   | `web/**`, `assets/**`                    |
| `eval/`          | eval-session  | `eval/**`                                |
| root config      | arch-session  | `CLAUDE.md`, `requirements.txt`, `.env.example`, `.gitignore`, `CONTRACT_CHANGES.md`, `README.md` |

**Parallel-session rule:** You own exactly one folder. Never edit files outside it.
If a contract must change, append the proposed change to `CONTRACT_CHANGES.md` and stop.

---

## Contracts (VERBATIM)

### Gesture Set

```python
POSE_SET = [
    "neutral", "fist", "index_point", "pinch", "tripod",
    "wrist_flex", "wrist_extend", "pronation", "supination"
]
```

### WindowEnvelope (JSON — the unit flowing through the system)

```json
{
  "t_ms": "<int>",
  "subject": "<str>",
  "true_gesture": "<str|null>",
  "spectrogram_png_b64": "<str>",
  "features": {
    "rms": ["<float x 12>"],
    "zero_crossings": ["<int x 12>"]
  }
}
```

### ClassifierResponse (what the model layer must return)

```json
{
  "gesture": "<str>",
  "confidence": "<float>",
  "drift_suspected": "<bool>",
  "notes": "<str>"
}
```

### WebSocket /stream

**Server → client:**

```json
{ "type": "classification", "t_ms": "<int>", "true_gesture": "<str|null>",
  "pred": "<ClassifierResponse>", "rolling_accuracy_20": "<float>",
  "cumulative_cost_usd": "<float>" }

{ "type": "recalibration_request", "target_gesture": "<str>",
  "confused_with": "<str>", "question": "<str>" }

{ "type": "recalibration_result", "gesture": "<str>",
  "acc_before": "<float>", "acc_after": "<float>" }
```

**Client → server:**

```json
{ "type": "control", "action": "play|pause|inject",
  "drift": "electrode_shift|sweat|channel_swap" }

{ "type": "recal_ack", "gesture": "<str>" }
```

### HTTP Endpoints

```
GET /history  → list of classifications
GET /stats    → {
    "confusion_matrix": "<10x10 ints>",
    "accuracy_series":  [{"t_ms": "<int>", "acc": "<float>"}],
    "total_tokens":     "<int>",
    "cost_usd":         "<float>"
}
```

### Python Stream API

```python
myosignal.stream(subject: str, drift_at_s: float | None) -> Iterator[WindowEnvelope]
```

---

## Run Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and fill in secrets
cp .env.example .env
# edit .env: set OPENROUTER_API_KEY=<your key>

# Start the FastAPI server (development)
uvicorn web.server:app --reload --port 8000

# Run tests
pytest eval/

# Run offline benchmark
python -m eval.bench

# Open dashboard (after server is running)
open assets/dashboard.html   # or xdg-open / start on Linux/Windows
```

---

## CONTRACT_CHANGES.md protocol

Any session needing to change a contract must:
1. Append the proposed delta to `CONTRACT_CHANGES.md` (do NOT edit `CLAUDE.md` directly).
2. Stop work and wait for the arch-session to review and merge.
