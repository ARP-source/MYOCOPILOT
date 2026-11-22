"""
myosignal.spectro, convert a sEMG window to a PNG spectrogram encoded as base64.

Uses short-time Fourier transform (STFT) on each channel, stacks the magnitude
spectrograms into an RGB image (channels mapped to colour), and returns the PNG
as a base64 string suitable for embedding in a WindowEnvelope.
"""

from __future__ import annotations
import base64
import numpy as np


def window_to_spectrogram_b64(
    window: np.ndarray,
    fs: int = 2000,
    nperseg: int = 64,
    noverlap: int = 32,
    image_size: tuple[int, int] = (224, 224),
) -> str:
    """
    Compute STFT spectrograms for each channel, compose into a PNG image,
    and return as a base64-encoded string (no data URI prefix).

    Parameters
    ----------
    window:
        np.ndarray of shape (window_samples, n_channels), float32.
    fs:
        Sampling frequency in Hz.
    nperseg:
        STFT window length in samples.
    noverlap:
        STFT overlap in samples.
    image_size:
        (width, height) of the output PNG in pixels.

    Returns
    -------
    Base64-encoded PNG string.
    """
    raise NotImplementedError
