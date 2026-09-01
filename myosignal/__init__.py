"""
myosignal — sEMG data loading, windowing, spectrogram generation, and live stream.

Public API (CLAUDE.md contract):
    stream(subject: str, drift_at_s: float | None) -> Iterator[WindowEnvelope]

Internal modules:
    loaders     — load_ninapro, load_emg2pose, make_synthetic_session
    preprocess  — bandpass, notch, z-score
    windows     — sliding-window segmentation + feature extraction
    spectrogram — per-channel STFT → 4×3 grid PNG → base64
    drift       — electrode_shift / sweat / channel_swap injection
    player      — SessionPlayer (real-time replay)
    stream      — public stream() function
    types       — WindowEnvelope, POSE_SET (shared with agent/)
"""

from myosignal.stream import stream
from myosignal.types import WindowEnvelope, POSE_SET
from myosignal.player import SessionPlayer

__all__ = ["stream", "WindowEnvelope", "POSE_SET", "SessionPlayer"]
