"""
web.server — FastAPI application.

Endpoints
---------
WS  /stream   : bidirectional WebSocket for live classification events
GET /history  : return list of all classification dicts for the session
GET /stats    : return confusion matrix, accuracy series, token and cost totals
"""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from myosignal import SessionPlayer
from myosignal.types import WindowEnvelope, POSE_SET
from agent import classify, check_drift, recalibrate
from agent.recalibrator import generate_interview_question
from agent.profile_store import load_profile
from web.session import Session

logger = logging.getLogger(__name__)

app = FastAPI(title="MyoCopilot API")

DEFAULT_SUBJECT = "DB2_s1"
ENVELOPE_QUEUE_MAXSIZE = 5

# Single active session — this demo serves one live dashboard at a time.
_current_session: Optional[Session] = None


def _pump_player(
    player: SessionPlayer,
    q: "queue.Queue[Optional[WindowEnvelope]]",
    stop: threading.Event,
    drift_at_s: float | None,
    drift_mode: str,
) -> None:
    """Run in a background thread: feed WindowEnvelopes into *q* at real-time
    pace. A full queue naturally throttles the player while paused. Puts a
    None sentinel once the stream ends or *stop* is set."""
    try:
        for envelope in player.stream(drift_at_s=drift_at_s, drift_mode=drift_mode):
            if stop.is_set():
                return
            q.put(envelope)
    finally:
        q.put(None)


@app.websocket("/stream")
async def ws_stream(websocket: WebSocket) -> None:
    """
    Bidirectional WebSocket endpoint.

    Server → client message types: "classification", "recalibration_request",
    "recalibration_result" (schemas in CLAUDE.md).

    Client → server message types: "control", "recal_ack".

    Behaviour
    ---------
    1. Accept connection and initialise a Session.
    2. Start the myosignal stream for the requested subject.
    3. For each WindowEnvelope:
       a. Call agent.classify() to get ClassifierResponse.
       b. Update DriftDetector; compute rolling_accuracy_20. Feed the
          window's features to the label free FeatureDriftDetector, which
          enrolls on the first 600 labeled windows (30 s) and then decides
          whether drift is happening.
       c. Emit "classification" message.
       d. If drift detected, emit "recalibration_request" and pause stream.
       e. On "recal_ack", run recalibrator and emit "recalibration_result".
    4. Handle "control" messages (play / pause / inject).
    """
    global _current_session

    await websocket.accept()

    params = websocket.query_params
    subject = params.get("subject", DEFAULT_SUBJECT)
    drift_at_s = float(params["drift_at_s"]) if "drift_at_s" in params else None
    drift_mode = params.get("drift_mode", "electrode_shift")

    session = Session(subject)
    _current_session = session

    player = SessionPlayer(subject=subject)
    envelope_q: "queue.Queue[Optional[WindowEnvelope]]" = queue.Queue(
        maxsize=ENVELOPE_QUEUE_MAXSIZE
    )
    stop_event = threading.Event()

    producer = threading.Thread(
        target=_pump_player,
        args=(player, envelope_q, stop_event, drift_at_s, drift_mode),
        daemon=True,
    )
    producer.start()

    recal_ack_event = asyncio.Event()
    awaiting_recal_gesture: Optional[str] = None

    async def receiver() -> None:
        nonlocal awaiting_recal_gesture
        try:
            while True:
                msg = await websocket.receive_json()
                msg_type = msg.get("type")
                if msg_type == "control":
                    action = msg.get("action")
                    if action == "play":
                        session.playing = True
                    elif action == "pause":
                        session.playing = False
                    elif action == "inject":
                        player.apply_drift(msg.get("drift", "electrode_shift"))
                elif msg_type == "recal_ack":
                    if msg.get("gesture") == awaiting_recal_gesture:
                        recal_ack_event.set()
        except WebSocketDisconnect:
            stop_event.set()

    receiver_task = asyncio.create_task(receiver())

    try:
        while True:
            if not session.playing:
                await asyncio.sleep(0.05)
                continue

            envelope = await asyncio.to_thread(envelope_q.get)
            if envelope is None:
                break  # stream ended

            history = [h["pred"] for h in session.history[-5:]]
            profile = load_profile(subject)
            pred = await classify(
                envelope, history=history, profile=profile, cost_tracker=session.cost
            )

            session.record_classification(
                t_ms=envelope["t_ms"],
                true_gesture=envelope["true_gesture"],
                pred_gesture=pred["gesture"],
                confidence=pred["confidence"],
                drift_suspected=pred["drift_suspected"],
                notes=pred["notes"],
            )

            session.observe_features(envelope)

            await websocket.send_json(
                {
                    "type": "classification",
                    "t_ms": envelope["t_ms"],
                    "true_gesture": envelope["true_gesture"],
                    "pred": pred,
                    "rolling_accuracy_20": session.detector.rolling_accuracy(),
                    "cumulative_cost_usd": session.cost.cost_usd,
                }
            )

            confusion_pair = check_drift(session.detector, session.feature_detector)
            if confusion_pair is not None:
                target_gesture, confused_with = confusion_pair
                question = await generate_interview_question(target_gesture, confused_with)

                awaiting_recal_gesture = target_gesture
                recal_ack_event.clear()

                await websocket.send_json(
                    {
                        "type": "recalibration_request",
                        "target_gesture": target_gesture,
                        "confused_with": confused_with,
                        "question": question,
                    }
                )

                await recal_ack_event.wait()

                result = await recalibrate(
                    subject,
                    target_gesture,
                    confused_with,
                    session.detector,
                    cost_tracker=session.cost,
                )
                awaiting_recal_gesture = None

                await websocket.send_json(
                    {
                        "type": "recalibration_result",
                        "gesture": result["gesture"],
                        "acc_before": result["acc_before"],
                        "acc_after": result["acc_after"],
                    }
                )
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        # Starlette raises RuntimeError (not WebSocketDisconnect) when a send
        # races a disconnect that happened mid-recalibration wait.
        logger.info("stream closed mid-send for subject=%s: %s", subject, exc)
    finally:
        stop_event.set()
        receiver_task.cancel()
        try:
            await receiver_task
        except asyncio.CancelledError:
            pass


@app.get("/history")
async def get_history() -> list[dict]:
    """
    Return the full classification history for the current session.

    Returns
    -------
    List of classification dicts (each matching the "classification" WS message
    schema minus the "type" key).
    """
    if _current_session is None:
        return []
    return _current_session.history


@app.get("/stats")
async def get_stats() -> dict:
    """
    Return aggregate statistics for the current session.

    Returns
    -------
    {
        "confusion_matrix": list[list[int]],   // 9x9 (one per POSE_SET entry)
        "accuracy_series":  list[{"t_ms": int, "acc": float}],
        "total_tokens":     int,
        "cost_usd":         float
    }
    """
    if _current_session is None:
        return {
            "confusion_matrix": [[0] * len(POSE_SET) for _ in POSE_SET],
            "accuracy_series": [],
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
    return {
        "confusion_matrix": _current_session.confusion_matrix_list(),
        "accuracy_series": _current_session.accuracy_series,
        "total_tokens": _current_session.cost.total_tokens,
        "cost_usd": _current_session.cost.cost_usd,
    }
