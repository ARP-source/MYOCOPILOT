"""
agent.recalibrator, conversational recalibration workflow.

When drift is detected the recalibrator:
1. Generates an interview question for the user via the LLM.
2. Waits for the user to perform the confused gesture (recal_ack from client).
3. Collects fresh spectrogram examples from the live stream.
4. Appends them to the per-user gesture profile via profile_store.
5. Returns a RecalibrationResult with before/after accuracy.
"""

from __future__ import annotations

import asyncio
import logging
import os

import httpx

from agent.types import ClassifierResponse, RecalibrationResult
from agent.drift_detector import DriftDetector
from agent.cost_tracker import CostTracker
from agent.profile_store import load_profile, append_examples
from agent.classifier import MODEL_ID, OPENROUTER_API_BASE
from myosignal import stream as myosignal_stream
from myosignal.types import WindowEnvelope

logger = logging.getLogger(__name__)

POST_RECAL_WINDOWS = 20

# Safety bound for _collect_target_gesture_windows: never scan more than this
# many windows of a fresh stream looking for the target gesture, so a subject
# whose stream never reproduces it returns early instead of scanning forever.
MAX_POST_RECAL_SCAN_WINDOWS = 5000

_QUESTION_SYSTEM_PROMPT = (
    "You are the conversational interface of a myoelectric prosthetic control "
    "system. The classifier is confusing two gestures. Write one short, "
    "friendly sentence asking the user to perform the target gesture a few "
    "times so the system can recalibrate. Respond with plain text only, no JSON."
)


async def generate_interview_question(
    target_gesture: str,
    confused_with: str,
) -> str:
    """
    Use the LLM to craft a user-friendly question asking the subject to
    perform *target_gesture* to help recalibrate the system.

    Parameters
    ----------
    target_gesture:
        The gesture the system is misclassifying.
    confused_with:
        The gesture the system is incorrectly predicting instead.

    Returns
    -------
    A natural-language question string suitable for display in the dashboard.
    """
    fallback = (
        f"I'm mixing up '{target_gesture}' with '{confused_with}' lately, "
        f"could you perform '{target_gesture}' a few times so I can recalibrate?"
    )

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if os.environ.get("MOCK") == "1" or not api_key:
        return fallback

    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{OPENROUTER_API_BASE}/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": MODEL_ID,
                    "messages": [
                        {"role": "system", "content": _QUESTION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                f"Target gesture: {target_gesture}\n"
                                f"Currently confused with: {confused_with}"
                            ),
                        },
                    ],
                    "temperature": 0.4,
                },
            )
            resp.raise_for_status()
            body = resp.json()
        text = body["choices"][0]["message"]["content"].strip()
        return text or fallback
    except (httpx.HTTPError, KeyError, IndexError, TypeError) as exc:
        logger.warning("generate_interview_question fell back to template: %s", exc)
        return fallback


async def collect_recalibration_samples(
    subject: str,
    gesture: str,
    n_samples: int = 10,
) -> list[str]:
    """
    Stream *n_samples* WindowEnvelopes for the given gesture from the live
    stream and return their spectrogram_png_b64 strings.

    Parameters
    ----------
    subject:
        Subject identifier used to look up the active stream.
    gesture:
        The gesture the subject is currently performing.
    n_samples:
        Number of fresh spectrogram examples to collect.

    Returns
    -------
    List of base64 PNG strings (one per example).
    """
    return await asyncio.to_thread(_collect_matching_examples, subject, gesture, n_samples)


def _collect_matching_examples(subject: str, gesture: str, n_samples: int) -> list[str]:
    """Blocking helper: scan a fresh myosignal stream for *gesture* windows."""
    examples: list[str] = []
    for envelope in myosignal_stream(subject):
        if envelope["true_gesture"] == gesture:
            examples.append(envelope["spectrogram_png_b64"])
            if len(examples) >= n_samples:
                break
    return examples


def _collect_matching_envelopes(subject: str, gesture: str, n_samples: int) -> list[WindowEnvelope]:
    """Blocking helper: like _collect_matching_examples, but returns the full
    confirmed WindowEnvelopes (not just their spectrogram PNGs) so callers
    can also enroll the underlying RMS/zero-crossing features, e.g.
    agent.mock_classifier's profile, which never looks at the spectrogram."""
    envelopes: list[WindowEnvelope] = []
    for envelope in myosignal_stream(subject):
        if envelope["true_gesture"] == gesture:
            envelopes.append(envelope)
            if len(envelopes) >= n_samples:
                break
    return envelopes


def _collect_target_gesture_windows(
    subject: str,
    gesture: str,
    n_windows: int,
    max_scan: int = MAX_POST_RECAL_SCAN_WINDOWS,
) -> list[WindowEnvelope]:
    """
    Blocking helper: scan a fresh myosignal stream for *n_windows* envelopes
    whose true_gesture equals *gesture*, ignoring every other gesture's
    windows along the way.

    A fresh stream restarts at t=0 of a brand-new synthetic/real session, and
    that session's first windows are not guaranteed (and for the synthetic
    generator, are actively unlikely) to be *gesture*, so grading
    post-recalibration accuracy on "whatever comes next" can silently score
    an unrelated gesture instead of the one that was just recalibrated. This
    filters explicitly so acc_after is always a true post-recalibration test
    of *gesture*.

    Parameters
    ----------
    subject:
        Subject identifier used to look up the active stream.
    gesture:
        The gesture that was just recalibrated, must be enrolled in the
        window's true_gesture, not merely present in a following window.
    n_windows:
        Number of matching windows to collect.
    max_scan:
        Safety bound on the total number of stream windows examined, so a
        stream that never reproduces *gesture* returns early (with fewer
        than *n_windows* results) instead of scanning forever.

    Returns
    -------
    List of WindowEnvelopes, all with true_gesture == gesture, in stream
    order. May contain fewer than *n_windows* entries if the stream is
    exhausted or *max_scan* is reached first.
    """
    out: list[WindowEnvelope] = []
    scanned = 0
    for envelope in myosignal_stream(subject):
        scanned += 1
        if envelope["true_gesture"] == gesture:
            out.append(envelope)
            if len(out) >= n_windows:
                break
        if scanned >= max_scan:
            break
    return out


async def recalibrate(
    subject: str,
    gesture: str,
    confused_with: str,
    detector: DriftDetector,
    *,
    cost_tracker: CostTracker | None = None,
) -> RecalibrationResult:
    """
    Run the full recalibration workflow for *gesture*:
    capture accuracy before, collect samples, update profile, return result.

    acc_after is measured over the next POST_RECAL_WINDOWS windows whose
    true_gesture equals *gesture* specifically (via
    _collect_target_gesture_windows, never just "whatever comes next" from
    the fresh stream, which restarts at t=0 of a brand-new session and is
    not guaranteed to replay *gesture* first). Those windows are classified
    with the refreshed reference profile and also fed into *detector* so the
    live session's rolling accuracy reflects the recalibration.

    Parameters
    ----------
    subject:
        Subject / user identifier.
    gesture:
        Gesture to recalibrate.
    confused_with:
        The gesture currently confused with *gesture*.
    detector:
        Active DriftDetector, used to snapshot accuracy before and after.
    cost_tracker:
        If given, records token usage from the post-recalibration classify
        calls (mirrors the cost_tracker accepted by agent.classifier.classify).

    Returns
    -------
    RecalibrationResult with gesture, acc_before, acc_after.
    """
    from agent import classify as _classify  # deferred: avoid import cycle with agent/__init__

    acc_before = detector.rolling_accuracy()

    confirmed_envelopes = await asyncio.to_thread(
        _collect_matching_envelopes, subject, gesture, 3
    )
    samples = [e["spectrogram_png_b64"] for e in confirmed_envelopes]
    if samples:
        append_examples(subject, gesture, samples)
    if confirmed_envelopes and os.environ.get("MOCK") == "1":
        from agent import mock_classifier

        mock_classifier.enroll_confirmed_samples(subject, gesture, confirmed_envelopes)

    profile = load_profile(subject)
    envelopes = await asyncio.to_thread(
        _collect_target_gesture_windows, subject, gesture, POST_RECAL_WINDOWS
    )
    if len(envelopes) < POST_RECAL_WINDOWS:
        logger.warning(
            "recalibrate(%s, %s): only found %d/%d post-recalibration '%s' windows "
            "within the %d-window scan bound",
            subject, gesture, len(envelopes), POST_RECAL_WINDOWS, gesture,
            MAX_POST_RECAL_SCAN_WINDOWS,
        )

    history: list[ClassifierResponse] = []
    for envelope in envelopes:
        pred = await _classify(
            envelope, history=history, profile=profile, cost_tracker=cost_tracker
        )
        history.append(pred)
        detector.update(envelope["true_gesture"], pred)

    acc_after = detector.rolling_accuracy()
    logger.info(
        "recalibrate(%s, %s): acc_before=%.3f acc_after=%.3f over %d target-gesture windows",
        subject, gesture, acc_before, acc_after, len(envelopes),
    )

    return {"gesture": gesture, "acc_before": acc_before, "acc_after": acc_after}
