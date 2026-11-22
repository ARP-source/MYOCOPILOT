"""
Full simulated session, per CLAUDE.md's testing note:

    clean session -> inject electrode_shift-style drift -> assert drift
    triggers -> recalibration completes -> recovery logged.

Also asserts the contract JSON shapes (ClassifierResponse, RecalibrationResult)
stay exactly as specified in CLAUDE.md.

Drift itself is simulated at the classification-outcome level (a run of
wrong predictions, standing in for what electrode_shift would cause a real
classifier to produce) rather than by replaying real sEMG through
myosignal.drift, that module is myosignal's contract to fulfil, not
agent's; agent only needs to react correctly to the (true, pred) stream it
is handed.
"""

from __future__ import annotations

import asyncio

import agent as agent_pkg
from agent.drift_detector import DriftDetector, check_drift
from agent.tests.conftest import make_envelope

POSE_SET = [
    "neutral", "fist", "index_point", "pinch", "tripod",
    "wrist_flex", "wrist_extend", "pronation", "supination",
]


def _assert_classifier_response_shape(resp: dict) -> None:
    assert set(resp.keys()) == {"gesture", "confidence", "drift_suspected", "notes"}
    assert resp["gesture"] in POSE_SET
    assert isinstance(resp["confidence"], (int, float))
    assert 0.0 <= resp["confidence"] <= 1.0
    assert isinstance(resp["drift_suspected"], bool)
    assert isinstance(resp["notes"], str)


def _assert_recalibration_result_shape(result: dict) -> None:
    assert set(result.keys()) == {"gesture", "acc_before", "acc_after"}
    assert result["gesture"] in POSE_SET
    assert isinstance(result["acc_before"], (int, float))
    assert isinstance(result["acc_after"], (int, float))


def test_full_session_clean_then_drift_then_recalibration(
    isolated_profiles_dir, monkeypatch
):
    detector = DriftDetector(window_size=20, accuracy_threshold=0.60)

    # --- Phase 1: clean session, classifier tracks the true gesture well ---
    clean_resp = {"gesture": "fist", "confidence": 0.92, "drift_suspected": False, "notes": "clean"}
    _assert_classifier_response_shape(clean_resp)
    for _ in range(60):
        detector.update("fist", clean_resp)

    assert check_drift(detector) is None
    assert detector.rolling_accuracy() == 1.0

    # --- Phase 2: electrode_shift-style drift, classifier starts confusing
    #     fist with pinch and loses confidence ---
    drift_resp = {"gesture": "pinch", "confidence": 0.3, "drift_suspected": True, "notes": "confused"}
    _assert_classifier_response_shape(drift_resp)
    for _ in range(20):
        detector.update("fist", drift_resp)

    pair = check_drift(detector)
    assert pair == ("fist", "pinch"), "drift must be detected and the confused pair identified"

    # --- Phase 3: recalibration, fresh 'fist' samples collected, profile
    #     updated, and the next 20 windows classify correctly again ---
    target_gesture, confused_with = pair
    fresh_samples = [make_envelope(t_ms=i * 50, true_gesture=target_gesture) for i in range(3)]
    post_recal_windows = [
        make_envelope(t_ms=2000 + i * 50, true_gesture=target_gesture) for i in range(20)
    ]
    streams = iter([fresh_samples, post_recal_windows])

    from agent import recalibrator

    def fake_stream(subject, drift_at_s=None, drift_mode="electrode_shift"):
        yield from next(streams)

    monkeypatch.setattr(recalibrator, "myosignal_stream", fake_stream)

    async def fake_classify(envelope, *, history=None, profile=None, cost_tracker=None, api_key=None):
        recovered = {
            "gesture": target_gesture,
            "confidence": 0.9,
            "drift_suspected": False,
            "notes": "recalibrated",
        }
        _assert_classifier_response_shape(recovered)
        return recovered

    monkeypatch.setattr(agent_pkg, "classify", fake_classify)

    result = asyncio.run(
        recalibrator.recalibrate("test_subj", target_gesture, confused_with, detector)
    )

    _assert_recalibration_result_shape(result)
    assert result["acc_before"] < 0.60, "acc_before should reflect the drifted rolling accuracy"
    assert result["acc_after"] == 1.0, "recovery: the next 20 windows all classify correctly"

    # --- Phase 4: recovery logged in the gesture profile ---
    from agent import profile_store

    profile = profile_store.load_profile("test_subj")
    assert len(profile["gestures"][target_gesture]["examples"]) == 3

    # Debounce holds, no immediate re-trigger right after recalibration.
    assert check_drift(detector) is None
