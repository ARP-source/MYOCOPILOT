"""
agent.classifier — call GLM-5.3-Flash via OpenRouter to classify a WindowEnvelope.

The spectrogram PNG is sent as a base64 image in the multimodal message.
The model is prompted to return a JSON ClassifierResponse.
"""

from __future__ import annotations

import json
import logging
import os
import re

import httpx
from dotenv import load_dotenv

from myosignal.types import WindowEnvelope, POSE_SET
from agent.types import ClassifierResponse
from agent.cost_tracker import CostTracker

load_dotenv()

logger = logging.getLogger(__name__)

MODEL_ID = "z-ai/glm-5.3-flash"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT_S = 30.0
MAX_REFERENCE_EXAMPLES = 3

SYSTEM_PROMPT = """You are the gesture-classification module of a myoelectric prosthetic control system.
You receive a spectrogram image of surface EMG from forearm electrode channels,
arranged in a 4x3 grid with channel numbers labeled. Higher intensity means more
muscle activation in that channel and frequency band. Classify the gesture into
EXACTLY ONE of: neutral, fist, index_point, pinch, tripod, wrist_flex, wrist_extend,
pronation, supination. Judge by the spatial pattern of energy across channels and its
frequency spread. If reference examples are attached, compare the pattern against
them first. Respond ONLY with JSON: {"gesture": <str>, "confidence": <0.0-1.0>,
"drift_suspected": <bool>, "notes": <one sentence>}. Set drift_suspected to true when
the pattern matches none of the classes well or clearly differs from the session's
established pattern. Be decisive — a prosthetic hand is waiting on you."""


def _select_reference_examples(
    profile: dict | None, max_examples: int = MAX_REFERENCE_EXAMPLES
) -> list[tuple[str, str]]:
    """Pick up to *max_examples* (gesture, png_b64) pairs from the most recently
    updated gestures in *profile*, one example per gesture."""
    if not profile:
        return []
    ranked = sorted(
        profile.get("gestures", {}).items(),
        key=lambda kv: kv[1].get("updated_at_ms", 0),
        reverse=True,
    )
    refs: list[tuple[str, str]] = []
    for gesture_name, data in ranked:
        examples = data.get("examples") or []
        if not examples:
            continue
        refs.append((gesture_name, examples[-1]))
        if len(refs) >= max_examples:
            break
    return refs


def _build_context_text(
    envelope: WindowEnvelope, history: list[ClassifierResponse] | None
) -> str:
    lines = [f"subject: {envelope['subject']}", f"t_ms: {envelope['t_ms']}"]
    if history:
        recent = ", ".join(
            f"{h['gesture']}({h['confidence']:.2f})" for h in history[-5:]
        )
        lines.append(f"last 5 predictions: {recent}")
    return "\n".join(lines)


def _build_messages(
    envelope: WindowEnvelope,
    history: list[ClassifierResponse] | None,
    profile: dict | None,
) -> list[dict]:
    content: list[dict] = [
        {"type": "text", "text": _build_context_text(envelope, history)}
    ]

    for gesture_name, png_b64 in _select_reference_examples(profile):
        content.append({"type": "text", "text": f"Reference — {gesture_name}:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/png;base64,{png_b64}"},
            }
        )

    content.append({"type": "text", "text": "Spectrogram to classify:"})
    content.append(
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{envelope['spectrogram_png_b64']}"
            },
        }
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": content},
    ]


def _extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    raise ValueError(f"could not extract JSON from model response: {text!r}")


def _coerce_response(data: dict) -> ClassifierResponse:
    gesture = data.get("gesture")
    if gesture not in POSE_SET:
        raise ValueError(f"gesture {gesture!r} not in POSE_SET")
    confidence = max(0.0, min(1.0, float(data.get("confidence", 0.0))))
    return {
        "gesture": gesture,
        "confidence": confidence,
        "drift_suspected": bool(data.get("drift_suspected", False)),
        "notes": str(data.get("notes", "")),
    }


async def _call_openrouter(messages: list[dict], api_key: str) -> tuple[dict, dict]:
    """POST one chat-completion request; return (parsed_json, usage_dict)."""
    async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_S) as client:
        resp = await client.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_ID,
                "messages": messages,
                "response_format": {"type": "json_object"},
                "temperature": 0.0,
            },
        )
        resp.raise_for_status()
        body = resp.json()

    choice_text = body["choices"][0]["message"]["content"]
    parsed = _extract_json(choice_text)
    usage = body.get("usage", {})
    return parsed, usage


async def classify(
    envelope: WindowEnvelope,
    *,
    history: list[ClassifierResponse] | None = None,
    profile: dict | None = None,
    cost_tracker: CostTracker | None = None,
    api_key: str | None = None,
) -> ClassifierResponse:
    """
    Send a WindowEnvelope to GLM-5.3-Flash and return a ClassifierResponse.

    The spectrogram PNG (base64) is embedded as a vision message.  The system
    prompt constrains the model to reply with valid JSON matching the
    ClassifierResponse schema.

    Parameters
    ----------
    envelope:
        A fully populated WindowEnvelope dict.
    history:
        Last few ClassifierResponses from this session, included as compact
        text context ("last 5 predictions").
    profile:
        The subject's gesture_profile dict (see agent.profile_store), used to
        attach up to MAX_REFERENCE_EXAMPLES few-shot reference images.
    cost_tracker:
        If given, records token usage from a successful call.
    api_key:
        Overrides OPENROUTER_API_KEY from the environment/.env file.

    Returns
    -------
    ClassifierResponse with gesture, confidence, drift_suspected, and notes.
    On hard failure (missing key, network error, or two bad model responses),
    returns {"gesture": <last prediction or "neutral">, "confidence": 0.05,
    "drift_suspected": True, "notes": "API fallback"} instead of raising, so
    the live demo never dies on stage.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    messages = _build_messages(envelope, history, profile)

    last_error: Exception | None = None
    if key:
        for attempt in range(2):
            try:
                parsed, usage = await _call_openrouter(messages, key)
                result = _coerce_response(parsed)
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError) as exc:
                last_error = exc
                logger.warning("classify attempt %d failed: %s", attempt + 1, exc)
                continue

            if cost_tracker is not None and usage:
                cost_tracker.record(
                    input_tokens=usage.get("prompt_tokens", 0),
                    output_tokens=usage.get("completion_tokens", 0),
                )
            logger.info(
                "classify usage subject=%s t_ms=%s prompt_tokens=%s completion_tokens=%s",
                envelope["subject"],
                envelope["t_ms"],
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
            return result
    else:
        last_error = RuntimeError("OPENROUTER_API_KEY is not set")

    logger.error("classify hard failure after retries: %s", last_error)
    last_gesture = history[-1]["gesture"] if history else "neutral"
    return {
        "gesture": last_gesture,
        "confidence": 0.05,
        "drift_suspected": True,
        "notes": "API fallback",
    }
