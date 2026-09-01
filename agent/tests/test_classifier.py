"""
Tests for agent.classifier — JSON extraction, response coercion, the
mocked-network happy path, retry-then-succeed, and the hard-failure fallback.
"""

from __future__ import annotations

import asyncio

import pytest

from agent import classifier
from agent.cost_tracker import CostTracker
from agent.tests.conftest import make_envelope


# ---------------------------------------------------------------------------
# _extract_json
# ---------------------------------------------------------------------------

class TestExtractJson:
    def test_plain_json(self):
        assert classifier._extract_json('{"gesture": "fist"}') == {"gesture": "fist"}

    def test_fenced_json(self):
        text = '```json\n{"gesture": "fist", "confidence": 0.9}\n```'
        assert classifier._extract_json(text) == {"gesture": "fist", "confidence": 0.9}

    def test_json_with_surrounding_prose(self):
        text = 'Sure, here you go: {"gesture": "pinch"} — hope that helps!'
        assert classifier._extract_json(text) == {"gesture": "pinch"}

    def test_unparseable_raises(self):
        with pytest.raises(ValueError):
            classifier._extract_json("not json at all")


# ---------------------------------------------------------------------------
# _coerce_response
# ---------------------------------------------------------------------------

class TestCoerceResponse:
    def test_valid_response_passthrough(self):
        result = classifier._coerce_response(
            {"gesture": "tripod", "confidence": 0.7, "drift_suspected": False, "notes": "ok"}
        )
        assert result == {
            "gesture": "tripod",
            "confidence": 0.7,
            "drift_suspected": False,
            "notes": "ok",
        }

    def test_confidence_clamped(self):
        result = classifier._coerce_response(
            {"gesture": "fist", "confidence": 5.0, "drift_suspected": True, "notes": ""}
        )
        assert result["confidence"] == 1.0

    def test_invalid_gesture_raises(self):
        with pytest.raises(ValueError):
            classifier._coerce_response(
                {"gesture": "finger_gun", "confidence": 0.5, "drift_suspected": False, "notes": ""}
            )


# ---------------------------------------------------------------------------
# classify() — mocked network
# ---------------------------------------------------------------------------

class TestClassifyHappyPath:
    def test_returns_coerced_response_and_records_cost(self, patch_openrouter):
        patch_openrouter(
            lambda payload: {
                "gesture": "fist",
                "confidence": 0.85,
                "drift_suspected": False,
                "notes": "clear fist pattern",
            }
        )
        envelope = make_envelope(t_ms=0, true_gesture="fist")
        cost_tracker = CostTracker()

        result = asyncio.run(
            classifier.classify(envelope, cost_tracker=cost_tracker, api_key="test-key")
        )

        assert result["gesture"] == "fist"
        assert result["confidence"] == 0.85
        assert result["drift_suspected"] is False
        assert cost_tracker.total_tokens == 540
        assert cost_tracker.cost_usd > 0

    def test_retries_once_then_succeeds(self, patch_openrouter):
        calls = {"n": 0}

        def responder(payload):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"gesture": "not_a_real_gesture"}  # triggers ValueError -> retry
            return {"gesture": "pinch", "confidence": 0.6, "drift_suspected": False, "notes": ""}

        patch_openrouter(responder)
        envelope = make_envelope(t_ms=100)

        result = asyncio.run(classifier.classify(envelope, api_key="test-key"))

        assert calls["n"] == 2
        assert result["gesture"] == "pinch"

    def test_reference_examples_attached_when_profile_has_history(self, patch_openrouter):
        seen_payloads = []

        def responder(payload):
            seen_payloads.append(payload)
            return {"gesture": "neutral", "confidence": 0.5, "drift_suspected": False, "notes": ""}

        patch_openrouter(responder)
        envelope = make_envelope(t_ms=0)
        profile = {
            "subject": "test_subj",
            "gestures": {
                "fist": {"examples": ["aaaa"], "updated_at_ms": 5},
                "pinch": {"examples": ["bbbb"], "updated_at_ms": 10},
            },
        }

        asyncio.run(classifier.classify(envelope, profile=profile, api_key="test-key"))

        content = seen_payloads[0]["messages"][1]["content"]
        image_parts = [c for c in content if c["type"] == "image_url"]
        # 2 reference images (one per gesture) + 1 query spectrogram.
        assert len(image_parts) == 3


# ---------------------------------------------------------------------------
# classify() — hard failure fallback
# ---------------------------------------------------------------------------

class TestClassifyFallback:
    def test_no_api_key_returns_fallback(self, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        envelope = make_envelope(t_ms=0)
        history = [{"gesture": "wrist_flex", "confidence": 0.8, "drift_suspected": False, "notes": ""}]

        result = asyncio.run(classifier.classify(envelope, history=history, api_key=None))

        assert result == {
            "gesture": "wrist_flex",
            "confidence": 0.05,
            "drift_suspected": True,
            "notes": "API fallback",
        }

    def test_persistent_bad_response_returns_fallback(self, patch_openrouter):
        patch_openrouter(lambda payload: {"gesture": "not_a_real_gesture"})
        envelope = make_envelope(t_ms=0)

        result = asyncio.run(classifier.classify(envelope, api_key="test-key"))

        assert result["notes"] == "API fallback"
        assert result["drift_suspected"] is True
        assert result["gesture"] == "neutral"  # no history -> default fallback gesture
