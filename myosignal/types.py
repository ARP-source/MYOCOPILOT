"""
Shared types for the myosignal package.
"""

from __future__ import annotations
from typing import TypedDict, Optional

POSE_SET: list[str] = [
    "neutral",
    "fist",
    "index_point",
    "pinch",
    "tripod",
    "wrist_flex",
    "wrist_extend",
    "pronation",
    "supination",
]


class WindowFeatures(TypedDict):
    """Per-channel features extracted from a 200 ms window."""

    rms: list[float]            # length 12, one per sEMG channel
    zero_crossings: list[int]   # length 12, one per sEMG channel


class WindowEnvelope(TypedDict):
    """
    The unit of data flowing through the system.

    t_ms               : wall-clock timestamp of the window centre (ms since epoch)
    subject            : dataset subject identifier (e.g. "DB2_s1")
    true_gesture       : ground-truth label from dataset, or null during live use
    spectrogram_png_b64: PNG spectrogram encoded as base64 string (no data URI prefix)
    features           : RMS and zero-crossing vectors, one entry per channel
    """

    t_ms: int
    subject: str
    true_gesture: Optional[str]
    spectrogram_png_b64: str
    features: WindowFeatures
