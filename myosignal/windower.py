"""
myosignal.windower — sliding-window segmentation and feature extraction.

Produces fixed-length overlapping windows from a raw sEMG array and
computes per-channel RMS and zero-crossing rate.
"""

from __future__ import annotations
import numpy as np
from myosignal.types import WindowFeatures

# Default window / stride in samples (at 2000 Hz: 200 ms window, 50 ms stride)
WINDOW_SAMPLES: int = 400
STRIDE_SAMPLES: int = 100
N_CHANNELS: int = 12


def extract_features(window: np.ndarray) -> WindowFeatures:
    """
    Compute per-channel RMS and zero-crossing count for a single window.

    Parameters
    ----------
    window:
        np.ndarray of shape (WINDOW_SAMPLES, N_CHANNELS), float32.

    Returns
    -------
    WindowFeatures dict with 'rms' (list of N_CHANNELS floats) and
    'zero_crossings' (list of N_CHANNELS ints).
    """
    raise NotImplementedError


def sliding_windows(
    emg: np.ndarray,
    window_samples: int = WINDOW_SAMPLES,
    stride_samples: int = STRIDE_SAMPLES,
) -> list[np.ndarray]:
    """
    Segment a raw sEMG array into overlapping windows.

    Parameters
    ----------
    emg:
        np.ndarray of shape (N, C), where N is total samples and C is channels.
    window_samples:
        Number of samples per window.
    stride_samples:
        Number of samples to advance between consecutive windows.

    Returns
    -------
    List of np.ndarray, each of shape (window_samples, C).
    """
    raise NotImplementedError
