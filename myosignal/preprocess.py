"""
myosignal.preprocess — signal conditioning for raw sEMG.

Pipeline (applied in order):
    1. 4th-order Butterworth bandpass  20–450 Hz  (filtfilt, zero phase)
    2. Optional 60 Hz notch            (IIR notch, filtfilt)
    3. Per-channel z-score normalisation using the first *baseline_s* seconds
       as the reference distribution.

All operations are applied per channel independently.  The input array is
never modified in-place; a float32 copy is always returned.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch


# ---------------------------------------------------------------------------
# Individual filter stages
# ---------------------------------------------------------------------------

def bandpass_filter(
    emg: np.ndarray,
    fs: int,
    lo: float = 20.0,
    hi: float = 450.0,
) -> np.ndarray:
    """
    Apply a 4th-order zero-phase Butterworth bandpass filter to every channel.

    Parameters
    ----------
    emg  : float32 array (n_samples, n_channels).
    fs   : sampling rate in Hz.
    lo   : lower cutoff in Hz (default 20 Hz).
    hi   : upper cutoff in Hz (default 450 Hz, below Nyquist for 2 kHz data).

    Returns
    -------
    Filtered float32 array of the same shape.
    """
    nyq = fs / 2.0
    lo_n = lo / nyq
    hi_n = min(hi / nyq, 0.999)   # guard against exceeding Nyquist
    b, a = butter(4, [lo_n, hi_n], btype="bandpass")
    out = np.empty_like(emg, dtype=np.float32)
    for ch in range(emg.shape[1]):
        out[:, ch] = filtfilt(b, a, emg[:, ch]).astype(np.float32)
    return out


def notch_filter(
    emg: np.ndarray,
    fs: int,
    freq: float = 60.0,
    q: float = 30.0,
) -> np.ndarray:
    """
    Apply a zero-phase IIR notch filter at *freq* Hz to every channel.

    Parameters
    ----------
    emg  : float32 array (n_samples, n_channels).
    fs   : sampling rate in Hz.
    freq : notch centre frequency in Hz (default 60 Hz, US mains).
    q    : quality factor controlling notch width (higher = narrower).

    Returns
    -------
    Filtered float32 array of the same shape.
    """
    b, a = iirnotch(freq / (fs / 2.0), q)
    out = np.empty_like(emg, dtype=np.float32)
    for ch in range(emg.shape[1]):
        out[:, ch] = filtfilt(b, a, emg[:, ch]).astype(np.float32)
    return out


# ---------------------------------------------------------------------------
# Z-score normalisation
# ---------------------------------------------------------------------------

def compute_baseline_stats(
    emg: np.ndarray,
    fs: int,
    baseline_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute per-channel mean and standard deviation from the first *baseline_s*
    seconds of the session.

    Parameters
    ----------
    emg        : float32 array (n_samples, n_channels).
    fs         : sampling rate in Hz.
    baseline_s : duration in seconds to use as the baseline window.

    Returns
    -------
    (mean, std) — each float32 array of shape (n_channels,).
    std values of zero are replaced with 1.0 to avoid division by zero.
    """
    n_baseline = min(int(baseline_s * fs), emg.shape[0])
    segment = emg[:n_baseline]
    mean = segment.mean(axis=0).astype(np.float32)
    std  = segment.std(axis=0).astype(np.float32)
    std[std == 0.0] = 1.0
    return mean, std


def zscore_normalize(
    emg: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    """
    Subtract *mean* and divide by *std* per channel.

    Parameters
    ----------
    emg  : float32 array (n_samples, n_channels).
    mean : float32 array (n_channels,).
    std  : float32 array (n_channels,).

    Returns
    -------
    Normalised float32 array of the same shape.
    """
    return ((emg - mean[np.newaxis, :]) / std[np.newaxis, :]).astype(np.float32)


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess(
    emg: np.ndarray,
    fs: int,
    notch: bool = False,
    notch_freq: float = 60.0,
    baseline_s: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the full preprocessing pipeline and return the conditioned array plus
    the baseline statistics (needed to normalise future windows consistently).

    Steps
    -----
    1. 4th-order Butterworth bandpass 20–450 Hz (filtfilt).
    2. (Optional) 60 Hz notch (filtfilt).
    3. Per-channel z-score using the first *baseline_s* seconds as reference.

    Parameters
    ----------
    emg        : raw float32 array (n_samples, n_channels).
    fs         : sampling rate in Hz.
    notch      : whether to apply the 60 Hz notch filter.
    notch_freq : notch centre frequency (default 60 Hz).
    baseline_s : baseline window duration for z-score statistics.

    Returns
    -------
    (emg_clean, baseline_mean, baseline_std)
        emg_clean     : preprocessed float32 array (n_samples, n_channels).
        baseline_mean : float32 array (n_channels,) — for downstream use.
        baseline_std  : float32 array (n_channels,) — for downstream use.
    """
    emg_f = bandpass_filter(emg, fs)
    if notch:
        emg_f = notch_filter(emg_f, fs, freq=notch_freq)
    mean, std = compute_baseline_stats(emg_f, fs, baseline_s)
    emg_z = zscore_normalize(emg_f, mean, std)
    return emg_z, mean, std
