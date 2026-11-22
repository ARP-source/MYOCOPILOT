"""
Shared fixtures for agent tests.

Two things make the real agent code slow or network-dependent in a test
run, and both are mocked here instead of using the real implementations:

    * agent.classifier talks to OpenRouter over the network, patched with
      an httpx.MockTransport so classify() still runs its real JSON
      extraction / coercion / retry logic against a scripted response.
    * agent.recalibrator streams fresh WindowEnvelopes from myosignal at
      real-time pace against a real (or synthetic) dataset, patched with an
      instant in-memory generator so tests don't block on wall-clock sleeps.
"""

from __future__ import annotations

import json
from typing import Callable

import httpx
import pytest


def make_envelope(
    t_ms: int,
    subject: str = "test_subj",
    true_gesture: str | None = "fist",
    rms: list[float] | None = None,
) -> dict:
    """A minimal WindowEnvelope matching the CLAUDE.md contract shape."""
    return {
        "t_ms": t_ms,
        "subject": subject,
        "true_gesture": true_gesture,
        "spectrogram_png_b64": "ZmFrZXBuZw==",
        "features": {
            "rms": rms if rms is not None else [0.1] * 12,
            "zero_crossings": [1] * 12,
        },
    }


def _openrouter_transport(responder: Callable[[dict], dict]) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        classifier_response = responder(payload)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(classifier_response)}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 40},
            },
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def patch_openrouter(monkeypatch):
    """
    Patch agent.classifier._call_openrouter so it routes through an
    httpx.MockTransport driven by a caller-supplied `responder(payload) ->
    ClassifierResponse dict` function, no real network access.
    """
    from agent import classifier

    def _patch(responder: Callable[[dict], dict]) -> None:
        transport = _openrouter_transport(responder)

        async def fake_call(messages: list[dict], api_key: str) -> tuple[dict, dict]:
            async with httpx.AsyncClient(
                transport=transport, base_url=classifier.OPENROUTER_API_BASE
            ) as client:
                resp = await client.post("/chat/completions", json={"messages": messages})
                resp.raise_for_status()
                body = resp.json()
            text = body["choices"][0]["message"]["content"]
            return classifier._extract_json(text), body.get("usage", {})

        monkeypatch.setattr(classifier, "_call_openrouter", fake_call)

    return _patch


@pytest.fixture
def patch_myosignal_stream(monkeypatch):
    """
    Patch agent.recalibrator's myosignal_stream reference with an instant,
    in-memory generator over a caller-supplied envelope list.
    """
    from agent import recalibrator

    def _patch(envelopes: list[dict]) -> None:
        def fake_stream(subject, drift_at_s=None, drift_mode="electrode_shift"):
            yield from envelopes

        monkeypatch.setattr(recalibrator, "myosignal_stream", fake_stream)

    return _patch


@pytest.fixture
def isolated_profiles_dir(monkeypatch, tmp_path):
    """Point agent.profile_store at a throwaway directory for this test."""
    from agent import profile_store

    monkeypatch.setattr(profile_store, "PROFILES_DIR", tmp_path / "gesture_profiles")
    return tmp_path / "gesture_profiles"
