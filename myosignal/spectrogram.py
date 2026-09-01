"""
myosignal.spectrogram — convert a sEMG window to a base64-encoded PNG spectrogram.

Design
------
- scipy.signal.stft with nperseg=256, noverlap=192 is run per channel.
- All 12 channels are arranged in a 4-column × 3-row grid.
- Colour scale: viridis, **fixed** dB range [ref_db − 60, ref_db].
  *ref_db* is the session-level maximum power in dB and must be computed once
  from the full session before streaming so every window is visually comparable.
- Output PNG is ≤ 512 px wide (height scales with the grid aspect ratio).
- The PNG bytes are base64-encoded without a data-URI prefix.
"""

from __future__ import annotations

import base64
import io
from typing import Optional

import numpy as np
from scipy.signal import stft

# Grid layout — must multiply to N_CHANNELS == 12
GRID_COLS: int = 4
GRID_ROWS: int = 3

# STFT parameters
NPERSEG: int = 256
NOVERLAP: int = 192

# Output width cap in pixels
MAX_WIDTH_PX: int = 512

# Fixed dynamic range in dB (bottom of colour scale = ref_db - DB_RANGE)
DB_RANGE: float = 60.0


# ---------------------------------------------------------------------------
# Session-level reference power
# ---------------------------------------------------------------------------

def compute_session_ref_db(
    emg: np.ndarray,
    fs: int,
    nperseg: int = NPERSEG,
    noverlap: int = NOVERLAP,
    sample_cap: int = 10_000,
) -> float:
    """
    Estimate the session maximum STFT power (in dB) from a representative sample.

    This value is used as the upper bound of the fixed colour scale so that all
    windows within a session are visually comparable.

    Parameters
    ----------
    emg        : float32 array (n_samples, n_channels) — entire or partial session.
    fs         : sampling rate in Hz.
    nperseg    : STFT window length.
    noverlap   : STFT overlap.
    sample_cap : maximum number of samples to use (speeds up large sessions).

    Returns
    -------
    ref_db : float — maximum dB power across all channels in the sample.
    """
    # Use a representative middle slice to avoid transients at the boundaries
    n = min(emg.shape[0], sample_cap)
    start = max(0, (emg.shape[0] - n) // 2)
    emg_sample = emg[start : start + n]

    max_power = 1e-20  # prevent log(0)
    for ch in range(emg_sample.shape[1]):
        _, _, Zxx = stft(emg_sample[:, ch], fs=fs, nperseg=nperseg, noverlap=noverlap)
        power = np.abs(Zxx) ** 2
        ch_max = float(power.max())
        if ch_max > max_power:
            max_power = ch_max

    return float(10.0 * np.log10(max_power + 1e-20))


# ---------------------------------------------------------------------------
# Single-window spectrogram
# ---------------------------------------------------------------------------

def window_to_png_b64(
    window: np.ndarray,
    fs: int,
    ref_db: float,
    nperseg: int = NPERSEG,
    noverlap: int = NOVERLAP,
    max_px_wide: int = MAX_WIDTH_PX,
) -> str:
    """
    Render a 4×3 grid of per-channel spectrograms into a PNG and return it
    as a base64 string.

    Parameters
    ----------
    window      : float32 array (window_samples, n_channels).
                  Fewer than 12 channels are zero-padded; more are truncated.
    fs          : sampling rate in Hz.
    ref_db      : session reference power in dB (from compute_session_ref_db).
                  Colour scale spans [ref_db − DB_RANGE, ref_db].
    nperseg     : STFT window length (default 256).
    noverlap    : STFT overlap (default 192).
    max_px_wide : maximum PNG width in pixels (default 512).

    Returns
    -------
    Base64-encoded PNG string (no data-URI prefix).
    """
    import matplotlib
    matplotlib.use("Agg")          # non-interactive backend; import after setting
    import matplotlib.pyplot as plt

    # Normalise channel count to 12
    n_ch = window.shape[1]
    if n_ch < 12:
        pad = np.zeros((window.shape[0], 12 - n_ch), dtype=np.float32)
        window = np.concatenate([window, pad], axis=1)
    elif n_ch > 12:
        window = window[:, :12]

    # dB colour limits
    vmax = ref_db
    vmin = ref_db - DB_RANGE

    # Figure dimensions: keep width ≤ max_px_wide at 100 dpi.
    # Subtract 2 px for sub-pixel rounding safety; avoid bbox_inches="tight"
    # so matplotlib honours the exact figure size.
    dpi = 100
    fig_w_in = (max_px_wide - 2) / dpi    # e.g. 5.10 inches → 510 px
    # Height proportional to grid aspect: ROWS/COLS
    fig_h_in = fig_w_in * (GRID_ROWS / GRID_COLS)

    fig, axes = plt.subplots(
        GRID_ROWS, GRID_COLS,
        figsize=(fig_w_in, fig_h_in),
        dpi=dpi,
    )
    fig.patch.set_facecolor("#0d0e10")

    for idx in range(12):
        row = idx // GRID_COLS
        col = idx % GRID_COLS
        ax  = axes[row, col]

        f_arr, t_arr, Zxx = stft(
            window[:, idx].astype(np.float64),
            fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
        )
        power_db = 10.0 * np.log10(np.abs(Zxx) ** 2 + 1e-20)

        ax.pcolormesh(
            t_arr, f_arr, power_db,
            vmin=vmin, vmax=vmax,
            cmap="viridis",
            shading="gouraud",
        )
        ax.set_facecolor("#0d0e10")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"ch{idx + 1}", fontsize=5, color="#8a8d94", pad=1)
        for spine in ax.spines.values():
            spine.set_edgecolor("#23252b")
            spine.set_linewidth(0.4)

    # Use subplots_adjust instead of tight_layout so the figure size is
    # respected exactly when saved without bbox_inches="tight".
    fig.subplots_adjust(left=0.01, right=0.99, top=0.97, bottom=0.01,
                        wspace=0.08, hspace=0.22)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("ascii")
