"""
myosignal.drift, deterministic electrode-artefact injection for sEMG windows.

Three drift modes (all deterministic given the same seed):

electrode_shift
    Simulates physical electrode movement.
    • Channels 3 and 7 (0-indexed) are swapped (cross-talk from adjacent muscles).
    • Channels 1 and 2 (0-indexed) are multiplied by 0.7 (partial contact loss).

sweat
    Simulates perspiration degrading the electrode-skin interface.
    • A linearly-increasing noise floor is added to all channels.
      Magnitude scales with *elapsed_s* (seconds since drift start).
    • A slow sinusoidal baseline wander (0.5 Hz) is superimposed.

channel_swap
    Simulates a loose connector or harness rotation.
    • All adjacent channel pairs are swapped: (0,1), (2,3), (4,5), …
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

DriftMode = Literal["electrode_shift", "sweat", "channel_swap"]


# ---------------------------------------------------------------------------
# Individual drift implementations
# ---------------------------------------------------------------------------

def _electrode_shift(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Permute channels 3 ↔ 7 and apply 0.7 gain on channels 1-2.

    Parameters
    ----------
    window : float32 array (n_samples, n_channels).
    rng    : numpy random generator (unused here �� kept for API consistency).

    Returns
    -------
    Corrupted float32 array of the same shape.
    """
    out = window.copy()
    n_ch = out.shape[1]

    # Swap channels 3 and 7 (only if both exist)
    if n_ch > 7:
        out[:, [3, 7]] = out[:, [7, 3]]

    # Attenuate channels 1 and 2
    if n_ch > 2:
        out[:, 1] *= 0.7
    if n_ch > 2:
        out[:, 2] *= 0.7

    return out.astype(np.float32)


def _sweat(
    window: np.ndarray,
    rng: np.random.Generator,
    elapsed_s: float = 0.0,
    fs: int = 2000,
) -> np.ndarray:
    """
    Add a linearly-rising noise floor and slow baseline wander.

    Parameters
    ----------
    window    : float32 array (n_samples, n_channels).
    rng       : numpy random generator for the noise component.
    elapsed_s : seconds elapsed since drift injection started.
                Higher values → more severe contamination.
    fs        : sampling rate in Hz (used to phase the wander correctly).

    Returns
    -------
    Corrupted float32 array of the same shape.
    """
    out  = window.copy()
    n_t  = out.shape[0]
    n_ch = out.shape[1]

    # Scale factor: linearly ramps from 0 at t=0 to 1 at t=60 s, then capped
    scale = min(elapsed_s / 60.0, 1.0)

    # Broadband noise floor, same amplitude on every channel
    noise_amp = scale * 0.15 * float(np.std(out) + 1e-8)
    noise = rng.standard_normal(out.shape).astype(np.float32) * noise_amp
    out += noise

    # Slow baseline wander (0.5 Hz sinusoid), correlated across channels
    t = np.arange(n_t, dtype=np.float32) / fs
    # Phase offset determined by elapsed_s so the wander is continuous
    wander_amp = scale * 0.08 * float(np.std(out) + 1e-8)
    phase = 2.0 * np.pi * 0.5 * elapsed_s   # continuous phase
    wander = wander_amp * np.sin(2.0 * np.pi * 0.5 * t + phase)
    out += wander[:, np.newaxis]             # broadcast to all channels

    return out.astype(np.float32)


def _channel_swap(window: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Swap all adjacent channel pairs: (0,1), (2,3), (4,5), …

    Parameters
    ----------
    window : float32 array (n_samples, n_channels).
    rng    : numpy random generator (unused, kept for API consistency).

    Returns
    -------
    Swapped float32 array of the same shape.
    """
    out = window.copy()
    n_ch = out.shape[1]
    for i in range(0, n_ch - 1, 2):
        out[:, [i, i + 1]] = out[:, [i + 1, i]]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Unified public interface
# ---------------------------------------------------------------------------

def inject_drift(
    window: np.ndarray,
    mode: DriftMode,
    rng: Optional[np.random.Generator] = None,
    elapsed_s: float = 0.0,
    fs: int = 2000,
) -> np.ndarray:
    """
    Apply a deterministic drift artefact to a sEMG window.

    Parameters
    ----------
    window    : float32 array (n_samples, n_channels).
    mode      : one of 'electrode_shift', 'sweat', 'channel_swap'.
    rng       : numpy random generator for reproducibility.
                If None, a default generator seeded with 0 is used.
    elapsed_s : seconds elapsed since drift started (used by 'sweat' mode
                to scale the contamination severity).
    fs        : sampling rate in Hz (used by 'sweat' for wander phasing).

    Returns
    -------
    Corrupted float32 array of the same shape as *window*.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if mode == "electrode_shift":
        return _electrode_shift(window, rng)
    elif mode == "sweat":
        return _sweat(window, rng, elapsed_s=elapsed_s, fs=fs)
    elif mode == "channel_swap":
        return _channel_swap(window, rng)
    else:
        raise ValueError(
            f"Unknown drift mode '{mode}'. "
            "Choose from: 'electrode_shift', 'sweat', 'channel_swap'."
        )
