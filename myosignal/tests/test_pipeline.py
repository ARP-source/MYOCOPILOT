"""
myosignal test suite.

Covers:
    - Window-count arithmetic
    - WindowEnvelope JSON schema compliance (against CLAUDE.md contract)
    - PNG bytes decode to valid image
    - Drift determinism (same seed → identical output)
    - Preprocess shapes and statistics
    - Session player yields correct types
"""

from __future__ import annotations

import base64
import io
import json

import numpy as np
import pytest

from myosignal.loaders import N_CHANNELS, make_synthetic_session
from myosignal.windows import (
    WINDOW_MS,
    STRIDE_MS,
    expected_window_count,
    segment,
)


# ---------------------------------------------------------------------------
# Window count
# ---------------------------------------------------------------------------

class TestWindowCount:
    def test_expected_count_formula(self):
        """expected_window_count() matches the actual segment() output length."""
        fs = 2000
        n_samples = fs * 3   # 3 seconds
        session = make_synthetic_session(n_seconds=3.0, fs=fs, seed=0)
        batches = segment(session.emg, session.labels, session.label_names,
                          session.reps, fs=fs)
        expected = expected_window_count(n_samples, fs)
        assert len(batches) == expected

    def test_window_size_samples(self):
        """Each batch window has the correct number of samples."""
        fs = 2000
        win_samples = int(round(WINDOW_MS * fs / 1000))
        session = make_synthetic_session(n_seconds=2.0, fs=fs, seed=1)
        batches = segment(session.emg, fs=fs)
        for b in batches:
            assert b.window.shape == (win_samples, N_CHANNELS), (
                f"Expected ({win_samples}, {N_CHANNELS}), got {b.window.shape}"
            )

    def test_stride_coverage(self):
        """Adjacent batches are offset by exactly STRIDE_MS samples."""
        fs = 2000
        stride_samples = int(round(STRIDE_MS * fs / 1000))
        session = make_synthetic_session(n_seconds=2.0, fs=fs, seed=2)
        batches = segment(session.emg, fs=fs)
        assert len(batches) >= 2
        for a, b in zip(batches, batches[1:]):
            assert b.sample_offset - a.sample_offset == stride_samples

    def test_short_signal_produces_zero_windows(self):
        """A signal shorter than one window produces an empty list."""
        fs = 2000
        win_samples = int(round(WINDOW_MS * fs / 1000))
        tiny_emg = np.zeros((win_samples - 1, N_CHANNELS), dtype=np.float32)
        assert segment(tiny_emg, fs=fs) == []

    def test_exact_one_window(self):
        """A signal exactly one window long produces exactly one batch."""
        fs = 2000
        win_samples = int(round(WINDOW_MS * fs / 1000))
        emg = np.random.default_rng(0).standard_normal(
            (win_samples, N_CHANNELS)
        ).astype(np.float32)
        batches = segment(emg, fs=fs)
        assert len(batches) == 1


# ---------------------------------------------------------------------------
# Feature shapes
# ---------------------------------------------------------------------------

class TestFeatures:
    def test_rms_shape(self, emg_clean):
        batches = segment(emg_clean, fs=2000)
        for b in batches:
            assert b.rms.shape == (N_CHANNELS,)

    def test_zero_crossings_shape(self, emg_clean):
        batches = segment(emg_clean, fs=2000)
        for b in batches:
            assert b.zero_crossings.shape == (N_CHANNELS,)

    def test_rms_nonnegative(self, emg_clean):
        batches = segment(emg_clean, fs=2000)
        for b in batches:
            assert np.all(b.rms >= 0)

    def test_zero_crossings_nonnegative(self, emg_clean):
        batches = segment(emg_clean, fs=2000)
        for b in batches:
            assert np.all(b.zero_crossings >= 0)

    def test_zero_signal_rms_is_zero(self):
        fs = 2000
        win_samples = int(round(WINDOW_MS * fs / 1000))
        emg = np.zeros((win_samples * 3, N_CHANNELS), dtype=np.float32)
        batches = segment(emg, fs=fs)
        for b in batches:
            assert np.allclose(b.rms, 0.0)


# ---------------------------------------------------------------------------
# WindowEnvelope schema (CLAUDE.md contract)
# ---------------------------------------------------------------------------

class TestEnvelopeSchema:
    def _make_one_envelope(self):
        from myosignal.player import SessionPlayer
        player = SessionPlayer(subject="schema_test", source="synthetic", seed=0)
        return next(player.stream())

    def test_required_keys(self):
        env = self._make_one_envelope()
        required = {"t_ms", "subject", "true_gesture", "spectrogram_png_b64", "features"}
        assert required.issubset(env.keys()), (
            f"Missing keys: {required - env.keys()}"
        )

    def test_t_ms_is_int(self):
        env = self._make_one_envelope()
        assert isinstance(env["t_ms"], int)

    def test_subject_is_str(self):
        env = self._make_one_envelope()
        assert isinstance(env["subject"], str)

    def test_true_gesture_is_str_or_none(self):
        env = self._make_one_envelope()
        assert env["true_gesture"] is None or isinstance(env["true_gesture"], str)

    def test_features_has_rms_and_zc(self):
        env = self._make_one_envelope()
        feats = env["features"]
        assert "rms" in feats and "zero_crossings" in feats

    def test_rms_length_is_12(self):
        env = self._make_one_envelope()
        assert len(env["features"]["rms"]) == N_CHANNELS

    def test_zero_crossings_length_is_12(self):
        env = self._make_one_envelope()
        assert len(env["features"]["zero_crossings"]) == N_CHANNELS

    def test_rms_elements_are_float(self):
        env = self._make_one_envelope()
        for v in env["features"]["rms"]:
            assert isinstance(v, float), f"rms element {v!r} is not float"

    def test_zero_crossings_elements_are_int(self):
        env = self._make_one_envelope()
        for v in env["features"]["zero_crossings"]:
            assert isinstance(v, int), f"zc element {v!r} is not int"

    def test_serialisable_to_json(self):
        env = self._make_one_envelope()
        # Should not raise
        dumped = json.dumps(env)
        reloaded = json.loads(dumped)
        assert reloaded["t_ms"] == env["t_ms"]


# ---------------------------------------------------------------------------
# PNG decode
# ---------------------------------------------------------------------------

class TestSpectrogramPng:
    def test_b64_decodes_to_png(self):
        from myosignal.player import SessionPlayer
        player = SessionPlayer(subject="png_test", source="synthetic", seed=0)
        env = next(player.stream())
        png_bytes = base64.b64decode(env["spectrogram_png_b64"])
        # PNG magic bytes: \x89PNG\r\n\x1a\n
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n", "Decoded bytes are not a valid PNG"

    def test_png_width_le_512(self):
        from PIL import Image
        from myosignal.player import SessionPlayer
        player = SessionPlayer(subject="width_test", source="synthetic", seed=0)
        env = next(player.stream())
        png_bytes = base64.b64decode(env["spectrogram_png_b64"])
        img = Image.open(io.BytesIO(png_bytes))
        assert img.width <= 512, f"PNG width {img.width} exceeds 512 px"

    def test_spectrogram_b64_is_nonempty_string(self):
        from myosignal.player import SessionPlayer
        player = SessionPlayer(subject="nonempty_test", source="synthetic", seed=0)
        env = next(player.stream())
        assert isinstance(env["spectrogram_png_b64"], str)
        assert len(env["spectrogram_png_b64"]) > 100


# ---------------------------------------------------------------------------
# Drift determinism
# ---------------------------------------------------------------------------

class TestDriftDeterminism:
    def _drift_outputs(self, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
        """Run inject_drift twice with the same seed, return both outputs."""
        from myosignal.drift import inject_drift
        fs = 2000
        win_samples = int(round(WINDOW_MS * fs / 1000))
        rng = np.random.default_rng(seed)
        window = rng.standard_normal((win_samples, N_CHANNELS)).astype(np.float32)
        out1 = inject_drift(window.copy(), mode=mode,
                            rng=np.random.default_rng(seed), elapsed_s=5.0, fs=fs)
        out2 = inject_drift(window.copy(), mode=mode,
                            rng=np.random.default_rng(seed), elapsed_s=5.0, fs=fs)
        return out1, out2

    def test_electrode_shift_deterministic(self):
        a, b = self._drift_outputs("electrode_shift", seed=7)
        assert np.array_equal(a, b)

    def test_sweat_deterministic(self):
        a, b = self._drift_outputs("sweat", seed=7)
        assert np.array_equal(a, b)

    def test_channel_swap_deterministic(self):
        a, b = self._drift_outputs("channel_swap", seed=7)
        assert np.array_equal(a, b)

    def test_different_seeds_differ_for_sweat(self):
        from myosignal.drift import inject_drift
        fs = 2000
        win_samples = int(round(WINDOW_MS * fs / 1000))
        window = np.random.default_rng(0).standard_normal(
            (win_samples, N_CHANNELS)
        ).astype(np.float32)
        out1 = inject_drift(window.copy(), "sweat",
                            rng=np.random.default_rng(1), elapsed_s=5.0, fs=fs)
        out2 = inject_drift(window.copy(), "sweat",
                            rng=np.random.default_rng(2), elapsed_s=5.0, fs=fs)
        assert not np.array_equal(out1, out2)

    def test_electrode_shift_channel_permutation(self):
        """Channels 3 and 7 are swapped; channels 1 and 2 are scaled by 0.7."""
        from myosignal.drift import inject_drift
        fs = 2000
        win_samples = int(round(WINDOW_MS * fs / 1000))
        window = np.eye(N_CHANNELS, dtype=np.float32).repeat(win_samples // N_CHANNELS + 1, axis=0)
        window = window[:win_samples]  # (win_samples, 12) with identity-like structure
        # Use a simple linearly-spaced window for easy channel comparison
        w = np.zeros((win_samples, N_CHANNELS), dtype=np.float32)
        for ch in range(N_CHANNELS):
            w[:, ch] = float(ch + 1)
        out = inject_drift(w, "electrode_shift", rng=np.random.default_rng(0))
        # Channel 3 and 7 should be swapped
        assert np.allclose(out[:, 3], w[:, 7]), "ch3 should equal original ch7"
        assert np.allclose(out[:, 7], w[:, 3]), "ch7 should equal original ch3"
        # Channels 1 and 2 should be 0.7× original
        assert np.allclose(out[:, 1], w[:, 1] * 0.7)
        assert np.allclose(out[:, 2], w[:, 2] * 0.7)

    def test_channel_swap_adjacent_pairs(self):
        """All adjacent pairs are swapped: (0,1), (2,3), (4,5), …"""
        from myosignal.drift import inject_drift
        fs = 2000
        win_samples = int(round(WINDOW_MS * fs / 1000))
        w = np.zeros((win_samples, N_CHANNELS), dtype=np.float32)
        for ch in range(N_CHANNELS):
            w[:, ch] = float(ch + 1)
        out = inject_drift(w, "channel_swap", rng=np.random.default_rng(0))
        for i in range(0, N_CHANNELS - 1, 2):
            assert np.allclose(out[:, i],     w[:, i + 1]), f"ch{i} should be original ch{i+1}"
            assert np.allclose(out[:, i + 1], w[:, i]),     f"ch{i+1} should be original ch{i}"


# ---------------------------------------------------------------------------
# Preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_output_shape_unchanged(self, emg_array):
        from myosignal.preprocess import preprocess
        out, _, _ = preprocess(emg_array, fs=2000)
        assert out.shape == emg_array.shape

    def test_output_dtype_float32(self, emg_array):
        from myosignal.preprocess import preprocess
        out, _, _ = preprocess(emg_array, fs=2000)
        assert out.dtype == np.float32

    def test_baseline_mean_near_zero(self, emg_array):
        """After z-scoring the baseline, the mean over the first 2 s ≈ 0."""
        from myosignal.preprocess import preprocess
        out, _, _ = preprocess(emg_array, fs=2000, baseline_s=2.0)
        baseline = out[:4000]  # first 2 s
        assert np.abs(baseline.mean()) < 0.5, (
            f"Baseline mean {baseline.mean():.4f} is unexpectedly large"
        )

    def test_notch_does_not_change_shape(self, emg_array):
        from myosignal.preprocess import preprocess
        out, _, _ = preprocess(emg_array, fs=2000, notch=True)
        assert out.shape == emg_array.shape


# ---------------------------------------------------------------------------
# Public stream() API
# ---------------------------------------------------------------------------

class TestPublicStreamAPI:
    def test_stream_returns_iterator(self):
        from myosignal import stream
        import inspect
        gen = stream("1", drift_at_s=None)
        assert hasattr(gen, "__next__"), "stream() must return an iterator"

    def test_first_envelope_from_stream(self):
        from myosignal import stream
        env = next(stream("api_test"))
        assert "t_ms" in env
        assert "spectrogram_png_b64" in env

    def test_stream_signature_matches_contract(self):
        """stream(subject, drift_at_s) — positional args work."""
        from myosignal import stream
        import inspect
        sig = inspect.signature(stream)
        params = list(sig.parameters.keys())
        assert params[0] == "subject"
        assert params[1] == "drift_at_s"
