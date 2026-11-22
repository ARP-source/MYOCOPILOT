"""
Tests for agent.recalibrator, interview question generation, sample
collection, and the full recalibrate() workflow (all against mocked
myosignal streaming and a mocked agent.classify).
"""

from __future__ import annotations

import asyncio

import agent as agent_pkg
from agent import recalibrator
from agent.drift_detector import DriftDetector
from agent.tests.conftest import make_envelope


class TestGenerateInterviewQuestion:
    def test_falls_back_to_template_in_mock_mode(self, monkeypatch):
        monkeypatch.setenv("MOCK", "1")
        question = asyncio.run(recalibrator.generate_interview_question("fist", "pinch"))
        assert "fist" in question
        assert "pinch" in question

    def test_falls_back_to_template_without_api_key(self, monkeypatch):
        monkeypatch.delenv("MOCK", raising=False)
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        question = asyncio.run(recalibrator.generate_interview_question("tripod", "pinch"))
        assert "tripod" in question


class TestCollectRecalibrationSamples:
    def test_filters_to_target_gesture(self, patch_myosignal_stream):
        envelopes = [
            make_envelope(t_ms=i * 50, true_gesture=g)
            for i, g in enumerate(["fist", "pinch", "fist", "fist", "pinch", "fist"])
        ]
        patch_myosignal_stream(envelopes)

        samples = asyncio.run(
            recalibrator.collect_recalibration_samples("subj", "fist", n_samples=3)
        )

        assert samples == [e["spectrogram_png_b64"] for e in envelopes if e["true_gesture"] == "fist"][:3]

    def test_returns_fewer_if_stream_exhausted(self, patch_myosignal_stream):
        patch_myosignal_stream([make_envelope(t_ms=0, true_gesture="pinch")])

        samples = asyncio.run(
            recalibrator.collect_recalibration_samples("subj", "fist", n_samples=3)
        )

        assert samples == []


class TestCollectTargetGestureWindows:
    """Regression coverage for the acc_after measurement bug: the old
    _collect_next_windows() graded whatever came next in a fresh stream,
    regardless of gesture. _collect_target_gesture_windows() must skip
    non-target windows and keep scanning (bounded) until it finds enough."""

    def test_skips_leading_non_target_windows(self, patch_myosignal_stream):
        # Target gesture 'fist' doesn't appear until window 40, well past
        # the first 30 windows of the fresh stream.
        other = [make_envelope(t_ms=i * 50, true_gesture="other_gesture") for i in range(40)]
        target = [make_envelope(t_ms=(40 + i) * 50, true_gesture="fist") for i in range(20)]
        patch_myosignal_stream(other + target)

        result = recalibrator._collect_target_gesture_windows("subj", "fist", 20)

        assert len(result) == 20
        assert all(e["true_gesture"] == "fist" for e in result)

    def test_bounded_by_max_scan_when_target_never_appears(self, monkeypatch):
        def infinite_other_gesture(subject, drift_at_s=None, drift_mode="electrode_shift"):
            i = 0
            while True:
                yield make_envelope(t_ms=i * 50, true_gesture="other_gesture")
                i += 1

        monkeypatch.setattr(recalibrator, "myosignal_stream", infinite_other_gesture)

        result = recalibrator._collect_target_gesture_windows(
            "subj", "fist", n_windows=20, max_scan=50
        )

        assert result == []  # never hangs, returns early once max_scan is reached


class TestRecalibrateGradesOnlyTargetWindows:
    """End-to-end: recalibrate() must compute acc_after only from
    target-gesture windows, even when the fresh post-recal stream leads
    with a long run of a different gesture."""

    def test_acc_after_ignores_leading_non_target_windows(self, isolated_profiles_dir, monkeypatch):
        target_gesture = "fist"

        confirmed = [make_envelope(t_ms=i * 50, true_gesture=target_gesture) for i in range(3)]
        # Fresh post-recal stream: 40 unrelated windows first, then 20 target windows.
        other = [make_envelope(t_ms=(1000 + i) * 50, true_gesture="pinch") for i in range(40)]
        post_target = [
            make_envelope(t_ms=(1040 + i) * 50, true_gesture=target_gesture) for i in range(20)
        ]
        streams = iter([confirmed, other + post_target])

        def fake_stream(subject, drift_at_s=None, drift_mode="electrode_shift"):
            yield from next(streams)

        monkeypatch.setattr(recalibrator, "myosignal_stream", fake_stream)

        async def fake_classify(envelope, *, history=None, profile=None, cost_tracker=None, api_key=None):
            # Always predicts the target gesture, correct for target windows,
            # deliberately wrong for the leading 'pinch' windows so the old
            # (unfiltered) behavior would have scored acc_after == 0.0.
            return {
                "gesture": target_gesture,
                "confidence": 0.9,
                "drift_suspected": False,
                "notes": "fake",
            }

        monkeypatch.setattr(agent_pkg, "classify", fake_classify)

        detector = DriftDetector(window_size=20)
        for _ in range(20):
            detector.update(target_gesture, {"gesture": "pinch", "confidence": 0.5, "drift_suspected": False, "notes": ""})

        result = asyncio.run(recalibrator.recalibrate("subj", target_gesture, "pinch", detector))

        # If acc_after had been measured over the leading 40 'pinch' windows
        # (or a mix), it would be < 1.0. Filtered correctly, all 20 scored
        # windows are the target gesture and all predicted correctly.
        assert result["acc_after"] == 1.0


class TestRecalibrate:
    def test_updates_profile_and_returns_result_shape(self, isolated_profiles_dir, monkeypatch):
        target_envelopes = [make_envelope(t_ms=i * 50, true_gesture="fist") for i in range(3)]
        post_envelopes = [make_envelope(t_ms=1000 + i * 50, true_gesture="fist") for i in range(20)]

        # collect_recalibration_samples() and the post-recal window collector
        # both stream from myosignal_stream, one call per phase, in order.
        streams = iter([target_envelopes, post_envelopes])

        def fake_stream(subject, drift_at_s=None, drift_mode="electrode_shift"):
            yield from next(streams)

        monkeypatch.setattr(recalibrator, "myosignal_stream", fake_stream)

        async def fake_classify(envelope, *, history=None, profile=None, cost_tracker=None, api_key=None):
            return {"gesture": "fist", "confidence": 0.9, "drift_suspected": False, "notes": "fake"}

        monkeypatch.setattr(agent_pkg, "classify", fake_classify)

        detector = DriftDetector(window_size=20)
        for _ in range(20):
            detector.update("fist", {"gesture": "pinch", "confidence": 0.5, "drift_suspected": False, "notes": ""})

        result = asyncio.run(recalibrator.recalibrate("subj", "fist", "pinch", detector))

        assert set(result.keys()) == {"gesture", "acc_before", "acc_after"}
        assert result["gesture"] == "fist"
        assert result["acc_before"] == 0.0  # all 20 prior predictions were wrong
        assert result["acc_after"] == 1.0  # all 20 post-recal predictions correct

        from agent import profile_store

        profile = profile_store.load_profile("subj")
        assert len(profile["gestures"]["fist"]["examples"]) == 3
