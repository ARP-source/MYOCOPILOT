"""
Tests for myosignal.player.apply_electrode_shift_stress — the severe,
deterministic whole-array electrode-shift stress transform used for
'electrode_shift' drift in the synthetic/replay path.
"""

from __future__ import annotations

import numpy as np
import pytest

from myosignal.loaders import N_CHANNELS
from myosignal.player import (
    apply_electrode_shift_stress,
    _ELECTRODE_SHIFT_PERMUTATION,
    _ELECTRODE_SHIFT_GAIN,
    _ELECTRODE_SHIFT_CONTACT_LOSS_CHANNELS,
    _ELECTRODE_SHIFT_CONTACT_LOSS_GAIN,
)

FS = 2000
WIN_SAMPLES = 500  # 250 ms @ 2000 Hz


def _make_window(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((WIN_SAMPLES, N_CHANNELS)).astype(np.float32)


class TestPermutation:
    def test_is_a_valid_permutation_of_all_12_channels(self):
        assert sorted(_ELECTRODE_SHIFT_PERMUTATION.tolist()) == list(range(N_CHANNELS))

    def test_has_no_fixed_points(self):
        """Every channel must move to a *different* position — a real
        full electrode-cap remap, not a partial/no-op shift."""
        for i, dest in enumerate(_ELECTRODE_SHIFT_PERMUTATION):
            assert dest != i, f"channel {i} maps to itself"


class TestGainVector:
    def test_gain_within_specified_bounds(self):
        assert len(_ELECTRODE_SHIFT_GAIN) == N_CHANNELS
        for g in _ELECTRODE_SHIFT_GAIN:
            assert 0.45 <= g <= 1.70

    def test_gain_has_meaningful_variation(self):
        assert max(_ELECTRODE_SHIFT_GAIN) - min(_ELECTRODE_SHIFT_GAIN) >= 1.0


class TestContactLoss:
    def test_exactly_two_channels_lose_contact(self):
        assert len(_ELECTRODE_SHIFT_CONTACT_LOSS_CHANNELS) == 2
        assert len(set(_ELECTRODE_SHIFT_CONTACT_LOSS_CHANNELS)) == 2

    def test_contact_loss_gain_is_severe_attenuation(self):
        assert 0.0 < _ELECTRODE_SHIFT_CONTACT_LOSS_GAIN < 0.3


class TestDeterminism:
    def test_same_input_gives_identical_output(self):
        window = _make_window(seed=1)
        out1 = apply_electrode_shift_stress(window.copy(), FS)
        out2 = apply_electrode_shift_stress(window.copy(), FS)
        assert np.array_equal(out1, out2)

    def test_deterministic_regardless_of_global_rng_state(self):
        """The transform must not depend on ambient numpy random state —
        it's fully self-seeded."""
        window = _make_window(seed=2)
        np.random.seed(0)
        out1 = apply_electrode_shift_stress(window.copy(), FS)
        np.random.seed(12345)
        out2 = apply_electrode_shift_stress(window.copy(), FS)
        assert np.array_equal(out1, out2)


class TestShapeAndChannelCoverage:
    def test_output_shape_matches_input(self):
        window = _make_window(seed=3)
        out = apply_electrode_shift_stress(window, FS)
        assert out.shape == window.shape
        assert out.dtype == np.float32

    def test_every_channel_is_affected(self):
        """A window with a distinct constant value per channel should come
        out with every channel changed from its original value (permutation
        + gain + wander/noise all touch every position)."""
        window = np.zeros((WIN_SAMPLES, N_CHANNELS), dtype=np.float32)
        for ch in range(N_CHANNELS):
            window[:, ch] = float(ch + 1)
        out = apply_electrode_shift_stress(window, FS)
        for ch in range(N_CHANNELS):
            assert not np.allclose(out[:, ch], window[:, ch]), f"channel {ch} unchanged"

    def test_permutation_moves_channel_identity(self):
        """Channel i's original content should show up (scaled) at its
        permuted destination, not at its original position."""
        window = np.zeros((WIN_SAMPLES, N_CHANNELS), dtype=np.float32)
        window[:, 0] = 5.0  # only channel 0 carries a strong, distinctive signal
        out = apply_electrode_shift_stress(window, FS)
        dest = int(_ELECTRODE_SHIFT_PERMUTATION[0])
        # The destination channel should carry roughly gain*5.0 in magnitude,
        # clearly larger than the (silent, only wander/noise) other channels.
        other_channels = [c for c in range(N_CHANNELS) if c != dest]
        dest_energy = float(np.abs(out[:, dest]).mean())
        other_energy = max(float(np.abs(out[:, c]).mean()) for c in other_channels)
        assert dest_energy > other_energy * 2
