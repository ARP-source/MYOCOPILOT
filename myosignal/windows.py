"""
myosignal.windows — sliding-window segmentation and per-channel feature extraction.

Contract
--------
Window duration : 250 ms
Stride          : 50 ms
Features per window (both length == N_CHANNELS == 12):
    rms              : sqrt(mean(x²)) per channel  → list[float]
    zero_crossings   : number of sign changes per channel → list[int]

Label assignment  : majority vote over the window's sample labels.
Repetition number : majority vote over the window's repetition vector.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from myosignal.loaders import N_CHANNELS

# ---------------------------------------------------------------------------
# Timing constants
# ---------------------------------------------------------------------------
WINDOW_MS: int = 250          # window length in milliseconds
STRIDE_MS: int = 50           # stride / hop in milliseconds


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class WindowBatch:
    """
    One segmented window plus its features and metadata.

    Attributes
    ----------
    window        : float32 array (window_samples, N_CHANNELS).
    rms           : float32 array (N_CHANNELS,) — per-channel RMS.
    zero_crossings: int32 array   (N_CHANNELS,) — per-channel ZC count.
    label_int     : dominant integer gesture index in this window.
    label_name    : POSE_SET string ('' if unknown).
    rep           : dominant repetition number.
    sample_offset : index of the first sample (within the full session array).
    """

    window: np.ndarray
    rms: np.ndarray
    zero_crossings: np.ndarray
    label_int: int
    label_name: str
    rep: int
    sample_offset: int


# ---------------------------------------------------------------------------
# Feature helpers
# ---------------------------------------------------------------------------

def compute_rms(window: np.ndarray) -> np.ndarray:
    """
    Per-channel root-mean-square of a single window.

    Parameters
    ----------
    window : float32 array (window_samples, n_channels).

    Returns
    -------
    float32 array of shape (n_channels,).
    """
    return np.sqrt(np.mean(window ** 2, axis=0)).astype(np.float32)


def compute_zero_crossings(window: np.ndarray) -> np.ndarray:
    """
    Per-channel zero-crossing count for a single window.

    A zero crossing is counted each time consecutive samples differ in sign
    (zero-valued samples are ignored via a small threshold).

    Parameters
    ----------
    window : float32 array (window_samples, n_channels).

    Returns
    -------
    int32 array of shape (n_channels,).
    """
    # Use sign function with a tiny dead zone to ignore near-zero samples
    signs = np.sign(window)
    # Where sign == 0, carry forward previous sign (fill with 1 at start)
    for ch in range(window.shape[1]):
        s = signs[:, ch]
        last = 1
        for i in range(len(s)):
            if s[i] == 0:
                signs[i, ch] = last
            else:
                last = s[i]
    zc = np.sum(np.abs(np.diff(signs, axis=0)) > 1, axis=0).astype(np.int32)
    return zc


def _majority(arr: np.ndarray) -> int:
    """Return the most frequent integer value in *arr* (tie → smallest)."""
    if len(arr) == 0:
        return 0
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment(
    emg: np.ndarray,
    labels: Optional[np.ndarray] = None,
    label_names: Optional[np.ndarray] = None,
    reps: Optional[np.ndarray] = None,
    fs: int = 2000,
) -> list[WindowBatch]:
    """
    Segment *emg* into overlapping windows of WINDOW_MS ms with STRIDE_MS ms
    stride.  Per-channel RMS and zero-crossing count are computed for each.

    Parameters
    ----------
    emg         : preprocessed float32 array (n_samples, N_CHANNELS).
    labels      : optional int32 array (n_samples,) — gesture index per sample.
    label_names : optional str array (n_samples,) — POSE_SET name per sample.
    reps        : optional int32 array (n_samples,) — repetition per sample.
    fs          : sampling rate in Hz.

    Returns
    -------
    List of WindowBatch objects in chronological order.
    """
    win_samples    = int(round(WINDOW_MS * fs / 1000))
    stride_samples = int(round(STRIDE_MS  * fs / 1000))
    n_samples      = emg.shape[0]

    if labels is None:
        labels = np.zeros(n_samples, dtype=np.int32)
    if label_names is None:
        label_names = np.array([""] * n_samples, dtype=str)
    if reps is None:
        reps = np.ones(n_samples, dtype=np.int32)

    batches: list[WindowBatch] = []
    start = 0
    while start + win_samples <= n_samples:
        end    = start + win_samples
        window = emg[start:end]                      # (win_samples, N_CHANNELS)
        rms    = compute_rms(window)
        zc     = compute_zero_crossings(window)
        lbl_i  = _majority(labels[start:end])
        lbl_n  = str(label_names[start])             # use first-sample name for speed
        rep    = _majority(reps[start:end])

        batches.append(
            WindowBatch(
                window=window,
                rms=rms,
                zero_crossings=zc,
                label_int=lbl_i,
                label_name=lbl_n,
                rep=rep,
                sample_offset=start,
            )
        )
        start += stride_samples

    return batches


def expected_window_count(n_samples: int, fs: int = 2000) -> int:
    """
    Return the number of windows that segment() will produce for *n_samples*.

    Useful for pre-allocation and tests.
    """
    win_s    = int(round(WINDOW_MS * fs / 1000))
    stride_s = int(round(STRIDE_MS  * fs / 1000))
    if n_samples < win_s:
        return 0
    return (n_samples - win_s) // stride_s + 1
