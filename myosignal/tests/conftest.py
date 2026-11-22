"""
Shared pytest fixtures for myosignal tests.
"""

from __future__ import annotations

import numpy as np
import pytest

from myosignal.loaders import N_CHANNELS, RawSession, make_synthetic_session

FS = 2000        # 2 kHz
DURATION_S = 5   # short enough to be fast in CI


@pytest.fixture(scope="session")
def raw_session() -> RawSession:
    """5-second synthetic RawSession, shared across all tests in the session."""
    return make_synthetic_session(n_seconds=DURATION_S, fs=FS, subject="test", seed=42)


@pytest.fixture(scope="session")
def emg_array(raw_session: RawSession) -> np.ndarray:
    """Raw EMG ndarray (n_samples, 12)."""
    return raw_session.emg


@pytest.fixture(scope="session")
def emg_clean(raw_session: RawSession) -> np.ndarray:
    """Preprocessed (bandpass + z-score) EMG ndarray."""
    from myosignal.preprocess import preprocess
    emg_z, _, _ = preprocess(raw_session.emg, raw_session.fs, notch=False, baseline_s=2.0)
    return emg_z
