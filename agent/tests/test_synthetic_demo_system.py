"""
System-level tests: the REAL myosignal.loaders.make_synthetic_session
structured synthetic signal, segmented through myosignal.windows, classified
end-to-end by agent.mock_classifier, with myosignal.player's severe
electrode-shift stress transform for drift — no controlled/hand-built
fixtures. This is the "does the actual demo story hold up" complement to
agent/tests/test_mock_classifier.py's controlled-fixture unit tests.

All data here is synthetic_demo_data: myosignal.loaders' deliberately
engineered structured EMG-like synthetic signals. This is a structured
synthetic electrode-shift STRESS TEST, not clinical validation of anything.
"""

from __future__ import annotations

import asyncio

import pytest

from myosignal.loaders import make_synthetic_session
from myosignal.windows import segment, compute_rms, compute_zero_crossings
from myosignal.player import apply_electrode_shift_stress
from agent import mock_classifier
from agent.drift_detector import DriftDetector, check_drift

DRIFT_FRACTION = 0.35
ROLLING_WINDOW = 20
SEEDS = (1, 2)


def _segment_session(subject: str, seed: int):
    raw = make_synthetic_session(subject=subject, seed=seed)
    batches = segment(
        raw.emg, labels=raw.labels, label_names=raw.label_names, reps=raw.reps, fs=raw.fs
    )
    return raw, batches


def _classify_clean(envelope: dict) -> dict:
    return asyncio.run(mock_classifier.classify(envelope))


class TestCleanSyntheticSession:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_post_bootstrap_accuracy_confidence_and_no_premature_trigger(self, seed):
        subject = f"sys_clean_{seed}"
        mock_classifier.reset_subject(subject)
        raw, batches = _segment_session(subject, seed)

        detector = DriftDetector(window_size=ROLLING_WINDOW, accuracy_threshold=0.60)
        correct = 0
        total = 0
        confidences: list[float] = []
        warm = True

        for i, b in enumerate(batches):
            true_g = b.label_name if b.label_name else None
            envelope = {
                "t_ms": i * 50,
                "subject": subject,
                "true_gesture": true_g,
                "spectrogram_png_b64": "",
                "features": {"rms": b.rms.tolist(), "zero_crossings": b.zero_crossings.tolist()},
            }
            pred = _classify_clean(envelope)
            if warm and "warm-up" not in pred["notes"]:
                warm = False

            if not warm:
                # Only monitor for drift once bootstrap has actually
                # finished — mirrors eval.demo.py / eval.run.py's own
                # pattern of not judging the classifier's forced warm-up
                # guesses as if they were live, calibrated predictions.
                detector.update(true_g, pred)
                assert check_drift(detector) is None, "clean session must never request recalibration"

                if true_g:
                    total += 1
                    confidences.append(pred["confidence"])
                    if pred["gesture"] == true_g:
                        correct += 1

        assert total > 0
        accuracy = correct / total
        mean_confidence = sum(confidences) / len(confidences)
        assert accuracy >= 0.80, f"seed={seed} post-bootstrap accuracy {accuracy:.3f} below 0.80"
        assert mean_confidence >= 0.70, f"seed={seed} mean confidence {mean_confidence:.3f} below 0.70"


class TestElectrodeShiftStress:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_drift_drops_below_60_percent_and_triggers_after_injection(self, seed):
        subject = f"sys_drift_{seed}"
        mock_classifier.reset_subject(subject)
        raw, batches = _segment_session(subject, seed)
        n = len(batches)
        drift_start = int(n * DRIFT_FRACTION)

        check_detector = DriftDetector(window_size=ROLLING_WINDOW, accuracy_threshold=0.60)
        trigger_pair = None
        trigger_window = None
        correct = 0
        total = 0

        # Clean phase: bootstrap + establish profiles, no monitoring needed
        # (mirrors production: the detector only starts once drift begins).
        for i in range(drift_start):
            b = batches[i]
            true_g = b.label_name if b.label_name else None
            envelope = {
                "t_ms": i * 50,
                "subject": subject,
                "true_gesture": true_g,
                "spectrogram_png_b64": "",
                "features": {"rms": b.rms.tolist(), "zero_crossings": b.zero_crossings.tolist()},
            }
            _classify_clean(envelope)

        # Drift phase: severe electrode-shift stress transform.
        for i in range(drift_start, n):
            b = batches[i]
            window = apply_electrode_shift_stress(b.window, raw.fs)
            rms = compute_rms(window)
            zc = compute_zero_crossings(window)
            true_g = b.label_name if b.label_name else None
            envelope = {
                "t_ms": i * 50,
                "subject": subject,
                "true_gesture": true_g,
                "spectrogram_png_b64": "",
                "features": {"rms": rms.tolist(), "zero_crossings": zc.tolist()},
            }
            pred = _classify_clean(envelope)
            check_detector.update(true_g, pred)
            if trigger_pair is None:
                trigger_pair = check_drift(check_detector)
                if trigger_pair is not None:
                    trigger_window = i

            if true_g:
                total += 1
                if pred["gesture"] == true_g:
                    correct += 1

        aggregate_accuracy = correct / total
        assert aggregate_accuracy < 0.60, (
            f"seed={seed} post-electrode-shift-stress aggregate accuracy "
            f"{aggregate_accuracy:.3f} did not drop below 0.60"
        )
        assert trigger_pair is not None, f"seed={seed}: DriftDetector never triggered after injection"
        assert trigger_window is not None and trigger_window >= drift_start, (
            "DriftDetector must trigger only after injection, not before"
        )
        # Debounce: an immediate re-check must not re-trigger.
        assert check_drift(check_detector) is None


class TestRecalibrationRecovery:
    @pytest.mark.parametrize("seed", SEEDS)
    def test_target_gesture_recovers_after_three_confirmed_samples(self, seed):
        subject = f"sys_recal_{seed}"
        mock_classifier.reset_subject(subject)
        raw, batches = _segment_session(subject, seed)
        n = len(batches)
        drift_start = int(n * DRIFT_FRACTION)

        detector = DriftDetector(window_size=ROLLING_WINDOW, accuracy_threshold=0.60)
        check_detector = DriftDetector(window_size=ROLLING_WINDOW, accuracy_threshold=0.60)

        for i in range(drift_start):
            b = batches[i]
            true_g = b.label_name if b.label_name else None
            envelope = {
                "t_ms": i * 50,
                "subject": subject,
                "true_gesture": true_g,
                "spectrogram_png_b64": "",
                "features": {"rms": b.rms.tolist(), "zero_crossings": b.zero_crossings.tolist()},
            }
            pred = _classify_clean(envelope)
            detector.update(true_g, pred)

        target_gesture = None
        i = drift_start
        while i < n and target_gesture is None:
            b = batches[i]
            window = apply_electrode_shift_stress(b.window, raw.fs)
            rms = compute_rms(window)
            zc = compute_zero_crossings(window)
            true_g = b.label_name if b.label_name else None
            envelope = {
                "t_ms": i * 50,
                "subject": subject,
                "true_gesture": true_g,
                "spectrogram_png_b64": "",
                "features": {"rms": rms.tolist(), "zero_crossings": zc.tolist()},
            }
            pred = _classify_clean(envelope)
            detector.update(true_g, pred)
            check_detector.update(true_g, pred)
            pair = check_drift(check_detector)
            if pair is not None:
                target_gesture = pair[0]
            i += 1

        assert target_gesture is not None, f"seed={seed}: drift never triggered a confused pair"

        # Recalibration: exactly 3 confirmed, CLEAN windows of target_gesture
        # (pulled from the original undrifted prefix, mirroring
        # agent.recalibrator's fresh-stream collection).
        clean_target = [bb for bb in batches[:drift_start] if bb.label_name == target_gesture]
        confirmed = clean_target[:3]
        assert len(confirmed) == 3, f"seed={seed}: fewer than 3 clean {target_gesture!r} samples available"
        confirmed_envelopes = [
            {
                "t_ms": 0,
                "subject": subject,
                "true_gesture": target_gesture,
                "spectrogram_png_b64": "",
                "features": {"rms": bb.rms.tolist(), "zero_crossings": bb.zero_crossings.tolist()},
            }
            for bb in confirmed
        ]
        mock_classifier.enroll_confirmed_samples(subject, target_gesture, confirmed_envelopes)

        acc_before = detector.rolling_accuracy()

        # Post-recalibration: next fresh, clean windows of target_gesture.
        post_recal = clean_target[3:23]
        assert len(post_recal) >= 15, f"seed={seed}: not enough post-recal {target_gesture!r} windows"
        target_correct = 0
        for bb in post_recal:
            envelope = {
                "t_ms": 0,
                "subject": subject,
                "true_gesture": target_gesture,
                "spectrogram_png_b64": "",
                "features": {"rms": bb.rms.tolist(), "zero_crossings": bb.zero_crossings.tolist()},
            }
            pred = _classify_clean(envelope)
            detector.update(target_gesture, pred)
            if pred["gesture"] == target_gesture:
                target_correct += 1

        target_accuracy = target_correct / len(post_recal)
        acc_after = detector.rolling_accuracy()

        assert target_accuracy >= 0.75, (
            f"seed={seed} target gesture {target_gesture!r} post-recalibration "
            f"accuracy {target_accuracy:.3f} below 0.75"
        )
        assert acc_after > acc_before, "overall rolling accuracy must show measurable recovery"


class TestNeverEchoesTrueGesture:
    def test_prediction_never_trivially_equals_true_gesture_via_shortcut(self):
        """Sanity check at the real-data-shape level: an envelope whose
        true_gesture is a gesture the classifier has never enrolled a
        profile for cannot be echoed back as the prediction."""
        subject = "sys_never_cheats"
        mock_classifier.reset_subject(subject)
        raw, batches = _segment_session(subject, seed=1)

        # Bootstrap on a couple of gestures only.
        bootstrapped = 0
        for b in batches:
            true_g = b.label_name if b.label_name else None
            envelope = {
                "t_ms": 0,
                "subject": subject,
                "true_gesture": true_g,
                "spectrogram_png_b64": "",
                "features": {"rms": b.rms.tolist(), "zero_crossings": b.zero_crossings.tolist()},
            }
            pred = _classify_clean(envelope)
            if "warm-up" not in pred["notes"]:
                bootstrapped += 1
            if bootstrapped > 5:
                break

        never_enrolled = "supination"
        snapshot = mock_classifier.get_profile_snapshot(subject)
        if never_enrolled not in snapshot["gestures"]:
            envelope = {
                "t_ms": 0,
                "subject": subject,
                "true_gesture": never_enrolled,
                "spectrogram_png_b64": "",
                "features": {"rms": [0.9] * 12, "zero_crossings": [50] * 12},
            }
            result = _classify_clean(envelope)
            assert result["gesture"] != never_enrolled
