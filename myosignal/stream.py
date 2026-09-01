"""
myosignal.stream — public streaming API (CLAUDE.md contract).

    myosignal.stream(subject: str, drift_at_s: float | None) -> Iterator[WindowEnvelope]

This module is the thin public façade over SessionPlayer.  All heavy lifting
(loading, preprocessing, windowing, spectrogram, drift) lives in player.py.
"""

from __future__ import annotations

from typing import Iterator

from myosignal.types import WindowEnvelope
from myosignal.player import SessionPlayer


def stream(
    subject: str,
    drift_at_s: float | None = None,
    drift_mode: str = "electrode_shift",
) -> Iterator[WindowEnvelope]:
    """
    Replay sEMG data for *subject* as a live stream of WindowEnvelopes.

    Matches the CLAUDE.md contract exactly:
        myosignal.stream(subject: str, drift_at_s: float | None) -> Iterator[WindowEnvelope]

    Parameters
    ----------
    subject    : Dataset subject identifier, e.g. "1", "DB2_s1", "emg2pose_u42".
                 The loader searches data/ for a matching NinaPro .mat or
                 emg2pose .h5 file.  If none is found, a synthetic session
                 is generated automatically so the pipeline is always runnable.
    drift_at_s : Seconds of clean replay before drift injection begins.
                 None → no drift injected.
    drift_mode : 'electrode_shift' | 'sweat' | 'channel_swap'.
                 Ignored when drift_at_s is None.

    Yields
    ------
    WindowEnvelope TypedDict at 1× real-time cadence.
    """
    player = SessionPlayer(subject=subject, source="auto")
    yield from player.stream(drift_at_s=drift_at_s, drift_mode=drift_mode)
