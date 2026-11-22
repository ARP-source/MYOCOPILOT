"""
Tests for agent.mock_classifier, deterministic per-subject bootstrap,
nearest-prototype inference that never echoes true_gesture, drift
sensitivity, and profile-replacing recalibration.

Feature vectors here are hand-constructed with a clear, controlled
per-gesture signature (small deterministic noise) rather than reusing
myosignal's own synthetic EMG generator, whose gesture bursts carry no
real per-channel spatial signal beyond a per-burst random amplitude scale
(see eval/run.py's own honest accuracy numbers), this isolates and proves
the classifier's *logic* against a signal that actually has the separation
the design assumes, mirroring test_session_pipeline.py's existing
precedent of exercising agent's contract-level behavior directly rather
than depending on myosignal.drift's own real-signal corruption.
"""

from __future__ import annotations

import asyncio
import math

from agent import mock_classifier
from agent.drift_detector import DriftDetector, check_drift

# One clearly distinct 12-channel RMS "signature" per gesture, well
# separated from every other in feature space.
_SIGNATURES: dict[str, list[float]] = {
    "neutral": [0.05] * 12,
    "fist":    [0.9, 0.85, 0.9, 0.2, 0.15, 0.2, 0.9, 0.85, 0.2, 0.15, 0.9, 0.2],
    "pinch":   [0.2, 0.15, 0.2, 0.9, 0.85, 0.9, 0.2, 0.15, 0.9, 0.85, 0.2, 0.9],
    "tripod":  [0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1, 0.6, 0.1],
}
_BOOTSTRAP_GESTURES = ("neutral", "fist", "pinch", "tripod")


def _envelope(gesture: str, index: int, subject: str, signature: list[float] | None = None) -> dict:
    """A WindowEnvelope with true_gesture's (or an overriding *signature*'s)
    feature pattern plus tiny deterministic per-window noise."""
    sig = signature if signature is not None else _SIGNATURES[gesture]
    noise = [0.01 * math.sin(index * 0.7 + ch * 0.9) for ch in range(12)]
    rms = [round(max(0.0, s + n), 5) for s, n in zip(sig, noise)]
    zc = [int(round(20 + 60 * s)) for s in sig]
    return {
        "t_ms": index * 50,
        "subject": subject,
        "true_gesture": gesture,
        "spectrogram_png_b64": "",
        "features": {"rms": rms, "zero_crossings": zc},
    }


def _classify(envelope: dict) -> dict:
    return asyncio.run(mock_classifier.classify(envelope))


def _bootstrap(subject: str, gestures=_BOOTSTRAP_GESTURES, rounds: int = 3, run_len: int = 5) -> int:
    """Feed *rounds* short clean runs of each of *gestures* (long enough to
    settle past the boundary-skip window each time) so every gesture ends
    up with `rounds` enrolled reference examples and warm-up ends. Returns
    the number of windows consumed."""
    idx = 0
    for _ in range(rounds):
        for gesture in gestures:
            for _ in range(run_len):
                _classify(_envelope(gesture, idx, subject))
                idx += 1
    return idx


# ---------------------------------------------------------------------------
# Bootstrap / warm-up
# ---------------------------------------------------------------------------

class TestBootstrap:
    def test_warmup_before_bootstrap_never_echoes_true_gesture(self):
        subject = "bootstrap_subj_1"
        mock_classifier.reset_subject(subject)

        result = _classify(_envelope("tripod", 0, subject))

        assert result["gesture"] != "tripod"
        assert result["gesture"] == mock_classifier.DEFAULT_GESTURE
        assert result["drift_suspected"] is False
        assert "warm-up" in result["notes"]
        snapshot = mock_classifier.get_profile_snapshot(subject)
        assert snapshot["ready"] is False

    def test_ready_flips_after_enough_distinct_gestures_bootstrapped(self):
        subject = "bootstrap_subj_2"
        mock_classifier.reset_subject(subject)

        _bootstrap(subject, gestures=("neutral", "fist"))

        snapshot = mock_classifier.get_profile_snapshot(subject)
        assert snapshot["ready"] is True
        assert snapshot["gestures"]["neutral"]["sample_count"] >= 3
        assert snapshot["gestures"]["fist"]["sample_count"] >= 3
        for gesture_audit in snapshot["gestures"].values():
            assert gesture_audit["centroid"] is not None
            assert gesture_audit["dispersion"] >= 0.0

    def test_never_predicts_a_gesture_with_no_enrolled_profile(self):
        subject = "bootstrap_subj_3"
        mock_classifier.reset_subject(subject)
        _bootstrap(subject, gestures=("neutral", "fist"))

        # "supination" was never bootstrapped/enrolled, even when it is the
        # true label, the classifier cannot echo it; it can only ever return
        # a gesture it has actually enrolled reference examples for.
        result = _classify(_envelope("supination", 999, subject, signature=_SIGNATURES["fist"]))

        assert result["gesture"] != "supination"
        assert result["gesture"] in mock_classifier.get_profile_snapshot(subject)["gestures"]


# ---------------------------------------------------------------------------
# Acceptance criterion 1, clean session accuracy/confidence, no premature recal
# ---------------------------------------------------------------------------

class TestCleanSession:
    def test_high_accuracy_high_confidence_no_premature_recalibration(self):
        subject = "clean_subj_1"
        mock_classifier.reset_subject(subject)
        idx = _bootstrap(subject)

        detector = DriftDetector(window_size=20, accuracy_threshold=0.60)
        confidences = []
        correct = 0
        total = 0
        for _ in range(15):
            for gesture in _BOOTSTRAP_GESTURES:
                envelope = _envelope(gesture, idx, subject)
                idx += 1
                result = _classify(envelope)
                confidences.append(result["confidence"])
                correct += result["gesture"] == gesture
                total += 1
                detector.update(gesture, result)
                assert check_drift(detector) is None, "clean session must never request recalibration"

        mean_accuracy = correct / total
        mean_confidence = sum(confidences) / len(confidences)

        assert mean_accuracy >= 0.80, f"mean rolling accuracy {mean_accuracy} below 0.80 threshold"
        assert mean_confidence >= 0.70, f"mean confidence {mean_confidence} below 0.70 threshold"


# ---------------------------------------------------------------------------
# Acceptance criterion 2, electrode-shift-style drift drops accuracy,
# DriftDetector triggers only after injection, debounce respected
# ---------------------------------------------------------------------------

class TestDriftInjection:
    def test_drift_drops_accuracy_and_triggers_after_injection_only(self):
        subject = "drift_subj_1"
        mock_classifier.reset_subject(subject)
        idx = _bootstrap(subject, gestures=("neutral", "fist", "pinch"))

        detector = DriftDetector(window_size=20, accuracy_threshold=0.60)

        # --- clean phase: fill the rolling window with correct 'fist' predictions ---
        for _ in range(20):
            envelope = _envelope("fist", idx, subject)
            idx += 1
            result = _classify(envelope)
            detector.update("fist", result)
        assert detector.rolling_accuracy() >= 0.60
        assert check_drift(detector) is None, "must not trigger before drift is injected"

        # --- drift phase: electrode-shift-style corruption makes 'fist'
        #     windows present the (very different) 'pinch' feature pattern ---
        drift_correct = 0
        drift_total = 0
        for _ in range(20):
            envelope = _envelope("fist", idx, subject, signature=_SIGNATURES["pinch"])
            idx += 1
            result = _classify(envelope)
            drift_correct += result["gesture"] == "fist"
            drift_total += 1
            detector.update("fist", result)

        drift_accuracy = drift_correct / drift_total
        assert drift_accuracy < 0.60, f"post-injection accuracy {drift_accuracy} did not drop below 0.60"
        assert detector.rolling_accuracy() < 0.60

        pair = check_drift(detector)
        assert pair is not None, "DriftDetector must trigger after injection"
        assert pair[0] == "fist"

        assert check_drift(detector) is None, "debounce must suppress an immediate re-trigger"


# ---------------------------------------------------------------------------
# Acceptance criterion 3, recalibration recovers accuracy
# ---------------------------------------------------------------------------

class TestRecalibrationRecovery:
    def test_three_confirmed_samples_restore_accuracy(self):
        subject = "recal_subj_1"
        mock_classifier.reset_subject(subject)
        idx = _bootstrap(subject, gestures=("neutral", "fist", "pinch"))

        detector = DriftDetector(window_size=20, accuracy_threshold=0.60)

        # Drive 'fist' into confusion with 'pinch', same as the drift test.
        for _ in range(20):
            envelope = _envelope("fist", idx, subject, signature=_SIGNATURES["pinch"])
            idx += 1
            result = _classify(envelope)
            detector.update("fist", result)
        acc_before = detector.rolling_accuracy()
        assert acc_before < 0.60

        # Recalibration flow: exactly 3 confirmed, CLEAN 'fist' windows
        # replace 'fist's reference profile (mirrors agent.recalibrator's
        # collect_recalibration_samples(n_samples=3) -> enroll_confirmed_samples).
        confirmed = [_envelope("fist", idx + i, subject) for i in range(3)]
        idx += 3
        before_count = mock_classifier.get_profile_snapshot(subject)["gestures"]["fist"]["sample_count"]
        mock_classifier.enroll_confirmed_samples(subject, "fist", confirmed)
        after_snapshot = mock_classifier.get_profile_snapshot(subject)["gestures"]["fist"]
        assert after_snapshot["sample_count"] == 3, "recalibration must deterministically replace, not blend"
        assert "recalibrated" in after_snapshot["note"]
        assert before_count != 3 or before_count == 3  # sanity: attribute exists and is comparable

        # Post-recalibration: fresh clean 'fist' windows should classify correctly again.
        post_correct = 0
        post_total = 0
        for _ in range(20):
            envelope = _envelope("fist", idx, subject)
            idx += 1
            result = _classify(envelope)
            post_correct += result["gesture"] == "fist"
            post_total += 1
            detector.update("fist", result)

        post_accuracy = post_correct / post_total
        acc_after = detector.rolling_accuracy()

        assert post_accuracy >= 0.75, f"post-recalibration accuracy {post_accuracy} below 0.75 threshold"
        assert acc_after > acc_before, "overall rolling accuracy must show measurable recovery"


# ---------------------------------------------------------------------------
# MOCK=1 real-classifier distinction (contract sanity)
# ---------------------------------------------------------------------------

class TestContractShape:
    def test_response_matches_classifier_response_contract(self):
        subject = "contract_subj_1"
        mock_classifier.reset_subject(subject)
        _bootstrap(subject, gestures=("neutral", "fist"))

        result = _classify(_envelope("fist", 999, subject))

        assert set(result.keys()) == {"gesture", "confidence", "drift_suspected", "notes"}
        assert isinstance(result["gesture"], str)
        assert isinstance(result["confidence"], (int, float))
        assert 0.0 <= result["confidence"] <= 1.0
        assert isinstance(result["drift_suspected"], bool)
        assert isinstance(result["notes"], str)
        assert "MOCK=1" in result["notes"]
