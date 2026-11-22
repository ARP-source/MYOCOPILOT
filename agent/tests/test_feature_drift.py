"""
Tests for agent.feature_drift.FeatureDriftDetector, its wiring into
check_drift() and web.session.Session, and a synthetic end to end check.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.drift_detector import DriftDetector, check_drift
from agent.feature_drift import (
    FeatureDriftDetector, calibrate_threshold, envelope_vector, persistence, rolling_mean,
)

N_CH = 12


def _session(seed: int = 0, n_reps: int = 4, per_run: int = 60, n_classes: int = 5):
    """Fake clean recording: each class has its own channel pattern, recorded
    in runs (class 0 rep 1, class 1 rep 1, ...). Returns X, y, rep, patterns."""
    rng = np.random.default_rng(seed)
    patterns = rng.uniform(0.3, 2.0, size=(n_classes, N_CH))
    zc = rng.uniform(5, 30, size=(n_classes, N_CH))
    X, y, rep = [], [], []
    for r in range(1, n_reps + 1):
        effort = rng.normal(1.0, 0.05)
        for c in range(n_classes):
            rms = patterns[c] * effort * np.exp(rng.normal(0, 0.1, size=(per_run, N_CH)))
            z = (zc[c] + rng.normal(0, 1.5, size=(per_run, N_CH))) / 50.0
            X.append(np.hstack([rms, z]))
            y += [c] * per_run
            rep += [r] * per_run
    return np.vstack(X), np.array(y), np.array(rep)


def _rotate(X):
    X = X.copy()
    X[:, :8] = np.roll(X[:, :8], 1, axis=1)
    X[:, N_CH:N_CH + 8] = np.roll(X[:, N_CH:N_CH + 8], 1, axis=1)
    return X


def _gain(X, g):
    X = X.copy()
    X[:, :N_CH] *= g
    return X


@pytest.fixture(scope="module")
def fitted():
    X, y, rep = _session(seed=0)
    det = FeatureDriftDetector(level_window=100).fit(X, y, rep)
    Xtest, ytest, _ = _session(seed=0, n_reps=6)
    return det, Xtest[len(X):]  # reps 5 and 6, not used for fitting


class TestHelpers:
    def test_rolling_mean_is_nan_until_full(self):
        out = rolling_mean(np.arange(5.0), 3)
        assert np.isnan(out[:2]).all()
        assert list(out[2:]) == [1.0, 2.0, 3.0]

    def test_persistence_k_of_n(self):
        f = np.array([1, 0, 1, 1, 0, 0, 0], bool)
        assert list(persistence(f, 2, 3)) == [False, False, True, True, True, False, False]

    def test_calibrate_threshold_hits_target(self):
        rng = np.random.default_rng(1)
        streams = [rng.normal(size=2000) for _ in range(3)]
        t = calibrate_threshold(streams, 1, 1, 0.05)
        rate = np.mean(np.concatenate(streams) > t)
        assert rate <= 0.05
        assert rate > 0.03

    def test_envelope_vector_layout(self):
        env = {"features": {"rms": [1.0] * N_CH, "zero_crossings": [50] * N_CH}}
        v = envelope_vector(env)
        assert len(v) == 2 * N_CH
        assert v[:N_CH] == [1.0] * N_CH and v[N_CH:] == [1.0] * N_CH


class TestFeatureDriftDetector:
    def test_update_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            FeatureDriftDetector().update([0.0] * 24)

    def test_cross_fit_calibration_used_with_groups(self, fitted):
        det, _ = fitted
        assert det.calibration == "cross_fit"
        assert np.isfinite(det.pattern_threshold) and np.isfinite(det.level_threshold)

    def test_in_sample_without_groups(self):
        X, y, _ = _session(seed=2)
        det = FeatureDriftDetector(level_window=100).fit(X, y)
        assert det.calibration == "in_sample"

    def test_quiet_on_clean_data(self, fitted):
        det, Xc = fitted
        r = det.run(Xc)
        assert r["alarm"].mean() <= 0.02

    def test_detects_rotation_through_pattern(self, fitted):
        det, Xc = fitted
        r = det.run(_rotate(Xc))
        assert r["pattern_alarm"][det.pattern_window + det.persist_n:].mean() > 0.9

    def test_detects_gain_through_level(self, fitted):
        det, Xc = fitted
        r = det.run(_gain(Xc, 0.6))
        assert r["level_alarm"][det.level_window:].mean() > 0.9

    def test_streaming_matches_vectorised(self, fitted):
        det, Xc = fitted
        X = np.vstack([Xc[:150], _rotate(Xc[150:300])])
        ref = det.run(X)["alarm"]
        det.reset_stream()
        streamed = [det.update(x) for x in X]
        det.reset_stream()
        assert streamed == list(ref)

    def test_reason(self, fitted):
        det, Xc = fitted
        det.reset_stream()
        for x in _rotate(Xc[:80]):
            det.update(x)
        assert det.reason == "pattern"
        det.reset_stream()
        assert det.reason is None

    def test_level_off_when_enrollment_too_short(self):
        X, y, rep = _session(seed=3, per_run=20)
        det = FeatureDriftDetector(level_window=200).fit(X, y, rep)
        assert det.level_threshold == float("inf")

    def test_min_level_threshold(self):
        X, y, rep = _session(seed=4)
        det = FeatureDriftDetector(level_window=100, min_level_threshold=5.0).fit(X, y, rep)
        assert det.level_threshold == 5.0

    def test_observe_enrolls_then_scores_without_labels(self):
        X, y, rep = _session(seed=5)
        det = FeatureDriftDetector(level_window=100, enroll_windows=len(X))
        for x, c in zip(X, y):
            assert det.observe(x, f"g{c}") is False
        assert det.fitted and det.calibration == "cross_fit"
        Xr = _rotate(X[:120])
        alarms = [det.observe(x, None) for x in Xr]
        assert any(alarms)

    def test_observe_ignores_unlabeled_windows_while_enrolling(self):
        det = FeatureDriftDetector(enroll_windows=10)
        for _ in range(50):
            det.observe([1.0] * 24, None)
        assert not det.fitted


class TestCheckDriftWithFeatureDetector:
    def _pred(self, g):
        return {"gesture": g, "confidence": 0.9, "drift_suspected": False, "notes": ""}

    def test_low_accuracy_alone_does_not_trigger_once_fitted(self, fitted):
        det, Xc = fitted
        det.reset_stream()
        for x in Xc[:200]:
            det.update(x)
        assert not det.alarm
        acc = DriftDetector(window_size=20)
        for _ in range(20):
            acc.update("fist", self._pred("pinch"))
        assert acc.drift_suspected()  # old rule would fire
        assert check_drift(acc, det) is None
        det.reset_stream()

    def test_feature_alarm_triggers_and_names_pair(self, fitted):
        det, Xc = fitted
        det.reset_stream()
        for x in _rotate(Xc[:80]):
            det.update(x)
        assert det.alarm
        acc = DriftDetector(window_size=20)
        for _ in range(15):
            acc.update("fist", self._pred("fist"))
        for _ in range(5):
            acc.update("fist", self._pred("pinch"))
        assert not acc.drift_suspected()
        assert check_drift(acc, det) == ("fist", "pinch")
        det.reset_stream()

    def test_falls_back_to_accuracy_rule_before_fit(self):
        acc = DriftDetector(window_size=20)
        for _ in range(20):
            acc.update("fist", self._pred("pinch"))
        assert check_drift(acc, FeatureDriftDetector()) == ("fist", "pinch")


class TestSessionWiring:
    def test_session_feeds_feature_detector(self):
        from web.session import Session

        s = Session("x")
        s.feature_detector = FeatureDriftDetector(enroll_windows=5)
        env = {"true_gesture": "fist", "features": {"rms": [1.0] * N_CH, "zero_crossings": [10] * N_CH}}
        for _ in range(4):
            s.observe_features(env)
        assert len(s.feature_detector._enroll_X) == 4


class TestSyntheticDemoSession:
    """Real synthetic demo signal and the player's electrode shift stress."""

    @pytest.mark.parametrize("seed", (1, 2))
    def test_quiet_when_clean_and_fires_after_electrode_shift(self, seed):
        from myosignal.loaders import make_synthetic_session
        from myosignal.windows import segment, compute_rms, compute_zero_crossings
        from myosignal.player import apply_electrode_shift_stress

        raw = make_synthetic_session(subject=f"fd_{seed}", seed=seed)
        batches = segment(raw.emg, labels=raw.labels, label_names=raw.label_names, reps=raw.reps, fs=raw.fs)

        def vec(rms, zc):
            return envelope_vector({"features": {"rms": list(rms), "zero_crossings": list(zc)}})

        clean = FeatureDriftDetector()
        clean_alarms = [clean.observe(vec(b.rms, b.zero_crossings), b.label_name or None) for b in batches]
        assert clean.fitted
        assert not any(clean_alarms)

        det = FeatureDriftDetector()
        first_alarm = None
        drift_from = None
        for i, b in enumerate(batches):
            if det.fitted:
                drift_from = i if drift_from is None else drift_from
                w = apply_electrode_shift_stress(b.window, raw.fs)
                v = vec(compute_rms(w), compute_zero_crossings(w))
            else:
                v = vec(b.rms, b.zero_crossings)
            if det.observe(v, b.label_name or None) and first_alarm is None:
                first_alarm = i
        assert first_alarm is not None and first_alarm >= drift_from
        assert (first_alarm - drift_from) * 0.05 < 3.0
