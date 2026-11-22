"""
myosignal.player, SessionPlayer: real-time sEMG replay with drift injection.

Usage
-----
    player = SessionPlayer("DB2_s1", source="ninapro")
    for envelope in player.stream(drift_at_s=30.0):
        process(envelope)

The player loads and preprocesses the full session once at construction,
then yields WindowEnvelope dicts at 1× real-time pace using wall-clock sleep.

Drift can be injected at a specified time offset or triggered at runtime
via apply_drift().  All three drift modes are deterministic given the seed.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from myosignal.loaders import (
    N_CHANNELS,
    RawSession,
    load_ninapro,
    load_emg2pose,
    make_synthetic_session,
)
from myosignal.preprocess import preprocess
from myosignal.windows import (
    segment,
    WindowBatch,
    WINDOW_MS,
    STRIDE_MS,
    compute_rms,
    compute_zero_crossings,
)
from myosignal.spectrogram import compute_session_ref_db, window_to_png_b64
from myosignal.drift import inject_drift, DriftMode
from myosignal.types import WindowEnvelope, POSE_SET

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path("data")

# ---------------------------------------------------------------------------
# Severe synthetic electrode-shift stress transform
# ---------------------------------------------------------------------------
# A deterministic, whole-array electrode-placement-shift model used ONLY for
# 'electrode_shift' drift in the synthetic/replay path (SessionPlayer.stream,
# and mirrored by eval.run's offline harness). This is a structured synthetic
# electrode-shift STRESS TEST, not a clinical or biomechanical model of real
# electrode drift.
#
# myosignal.drift.inject_drift's 'electrode_shift' (a 2-channel swap + a
# 2-channel gain cut) models a mild, localized shift. This transform instead
# models a severe full electrode-cap remap: every one of the 12 channels is
# displaced to a different physical position (a fixed derangement, no
# channel maps to itself), rescaled by a fixed per-channel gain, given a
# small channel-dependent baseline-wander/contact-noise artifact, and two
# channels additionally lose most contact. Unlike inject_drift's other modes
# (sweat, channel_swap), this one is intentionally left untouched by this
# transform and still goes through inject_drift as before.
#
# Every parameter below is a fixed constant (no external RNG, no session
# seed) so the transform is fully deterministic on its own, the same input
# window always produces the same output.

# Derangement of channel indices 0-11: perm[i] = (5*i + 3) mod 12. gcd(5,12)=1
# so this is a valid bijection; solving 4i ≡ 9 (mod 12) for a fixed point has
# no solution since gcd(4,12)=4 does not divide 9, so no channel maps to
# itself.
_ELECTRODE_SHIFT_PERMUTATION = np.array([3, 8, 1, 6, 11, 4, 9, 2, 7, 0, 5, 10])

# Fixed per-channel gain applied after the permutation (0.45x-1.70x).
_ELECTRODE_SHIFT_GAIN = np.array(
    [1.70, 0.45, 1.25, 0.60, 1.50, 0.80, 1.10, 0.50, 1.65, 0.70, 1.35, 0.55],
    dtype=np.float32,
)

# Channels that additionally lose most contact (near-total attenuation).
_ELECTRODE_SHIFT_CONTACT_LOSS_CHANNELS = (4, 9)
_ELECTRODE_SHIFT_CONTACT_LOSS_GAIN = 0.12

# Small, fixed, channel-dependent baseline wander / contact-noise texture.
_ELECTRODE_SHIFT_WANDER_HZ = 0.8
_ELECTRODE_SHIFT_WANDER_AMPLITUDE = 0.8
_ELECTRODE_SHIFT_NOISE_AMPLITUDE = 1.5
_ELECTRODE_SHIFT_NOISE_SEED = 20260831  # fixed, independent of session/subject seed


def apply_electrode_shift_stress(window: np.ndarray, fs: int) -> np.ndarray:
    """
    Apply the severe, deterministic electrode-placement-shift stress
    transform to one preprocessed sEMG window, see the module comment
    above _ELECTRODE_SHIFT_PERMUTATION.

    Parameters
    ----------
    window : float32 array (n_samples, n_channels).
    fs     : sampling rate in Hz (used to phase the baseline wander).

    Returns
    -------
    Corrupted float32 array of the same shape.
    """
    n_t, n_ch = window.shape

    # 1. Deterministic full-channel permutation (electrode cap remapped).
    perm = _ELECTRODE_SHIFT_PERMUTATION[:n_ch] % n_ch
    out = np.zeros_like(window)
    out[:, perm] = window

    # 2. Fixed per-channel gain (0.45x-1.70x).
    out *= _ELECTRODE_SHIFT_GAIN[:n_ch]

    # 3. Small channel-dependent baseline wander + contact noise, fully
    #    self-seeded so this step alone is deterministic regardless of caller state.
    rng = np.random.default_rng(_ELECTRODE_SHIFT_NOISE_SEED)
    t = np.arange(n_t, dtype=np.float32) / fs
    channel_scale = float(np.std(out) + 1e-8)
    for ch in range(n_ch):
        phase = (ch / n_ch) * 2.0 * np.pi
        wander = (
            _ELECTRODE_SHIFT_WANDER_AMPLITUDE
            * channel_scale
            * np.sin(2.0 * np.pi * _ELECTRODE_SHIFT_WANDER_HZ * t + phase)
        ).astype(np.float32)
        noise = (
            rng.standard_normal(n_t).astype(np.float32)
            * _ELECTRODE_SHIFT_NOISE_AMPLITUDE
            * channel_scale
        )
        out[:, ch] += wander + noise

    # 4. Two channels additionally lose most contact.
    for ch in _ELECTRODE_SHIFT_CONTACT_LOSS_CHANNELS:
        if ch < n_ch:
            out[:, ch] *= _ELECTRODE_SHIFT_CONTACT_LOSS_GAIN

    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Source auto-detection
# ---------------------------------------------------------------------------

def _find_data_file(subject: str, source: str) -> Optional[Path]:
    """
    Search DATA_DIR for a file matching *subject* and *source* format.

    Naming conventions checked (in priority order):
        ninapro  : S{n}_E1_A1.mat, S{n}_E2_A1.mat, {subject}.mat
        emg2pose : {subject}.h5, {subject}.hdf5
        synthetic: returns None (no file needed)
    """
    if not DATA_DIR.exists():
        return None

    if source in ("ninapro", "auto"):
        for suffix in (".mat",):
            # Try common NinaPro naming patterns
            for stem in (
                f"S{subject}_E1_A1",
                f"S{subject}_E2_A1",
                f"S{subject}_E3_A1",
                subject,
            ):
                p = DATA_DIR / f"{stem}{suffix}"
                if p.exists():
                    return p
        # Broader glob
        mats = list(DATA_DIR.glob(f"*{subject}*.mat"))
        if mats:
            return mats[0]

    if source in ("emg2pose", "auto"):
        for suffix in (".h5", ".hdf5"):
            p = DATA_DIR / f"{subject}{suffix}"
            if p.exists():
                return p
        h5s = list(DATA_DIR.glob(f"*{subject}*.h5")) + list(DATA_DIR.glob(f"*{subject}*.hdf5"))
        if h5s:
            return h5s[0]

    return None


def _detect_source(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".mat":
        return "ninapro"
    if suffix in (".h5", ".hdf5"):
        return "emg2pose"
    return "unknown"


# ---------------------------------------------------------------------------
# SessionPlayer
# ---------------------------------------------------------------------------

class SessionPlayer:
    """
    Load, preprocess, and replay one sEMG session as a stream of WindowEnvelopes.

    Parameters
    ----------
    subject   : subject identifier string (e.g. '1', 'DB2_s1', 'emg2pose_u42').
    source    : 'ninapro' | 'emg2pose' | 'synthetic' | 'auto'.
                'auto' searches data/ for any matching file.
    data_dir  : root directory for dataset files (default 'data/').
    seed      : RNG seed for drift determinism.
    notch     : enable 60 Hz notch filter in preprocessing.
    baseline_s: seconds of baseline for z-score normalisation.
    """

    def __init__(
        self,
        subject: str,
        source: str = "auto",
        data_dir: Optional[Path] = None,
        seed: int = 0,
        notch: bool = False,
        baseline_s: float = 30.0,
    ) -> None:
        self.subject = subject
        self.source  = source
        self.seed    = seed
        self._data_dir = Path(data_dir) if data_dir else DATA_DIR
        self._notch    = notch
        self._baseline_s = baseline_s

        self._session: Optional[RawSession]    = None
        self._windows: Optional[list[WindowBatch]] = None
        self._ref_db:  float                   = 0.0
        self._drift_mode: Optional[DriftMode]  = None
        self._rng = np.random.default_rng(seed)

        self._load_and_prepare()

    # ------------------------------------------------------------------
    # Loading / preprocessing
    # ------------------------------------------------------------------

    def _load_and_prepare(self) -> None:
        """Load raw data, preprocess, segment, and compute spectrogram ref."""
        path = _find_data_file(self.subject, self.source)

        if path is None:
            if self.source not in ("synthetic", "auto"):
                raise FileNotFoundError(
                    f"No data file found for subject='{self.subject}' "
                    f"source='{self.source}' in {self._data_dir}. "
                    "Place NinaPro .mat or emg2pose .h5 files under data/, "
                    "or use source='synthetic' for a demo."
                )
            # Fall through to synthetic
            detected = "synthetic"
        else:
            detected = _detect_source(path)

        if detected == "ninapro":
            raw = load_ninapro(path, subject=self.subject)
        elif detected == "emg2pose":
            raw = load_emg2pose(path, subject=self.subject)
        else:
            raw = make_synthetic_session(subject=self.subject, seed=self.seed)

        self.source  = raw.source   # update to actual source
        self._session = raw

        # --- Preprocess ---
        emg_clean, _, _ = preprocess(
            raw.emg, raw.fs,
            notch=self._notch,
            baseline_s=self._baseline_s,
        )

        # --- Segment ---
        self._windows = segment(
            emg_clean,
            labels=raw.labels,
            label_names=raw.label_names,
            reps=raw.reps,
            fs=raw.fs,
        )

        # --- Spectrogram reference power ---
        self._ref_db = compute_session_ref_db(emg_clean, raw.fs)

    # ------------------------------------------------------------------
    # Drift control
    # ------------------------------------------------------------------

    def apply_drift(self, kind: DriftMode) -> None:
        """
        Activate drift injection of the given *kind* from the next window onward.

        Parameters
        ----------
        kind : 'electrode_shift' | 'sweat' | 'channel_swap'.
        """
        self._drift_mode = kind
        # Reset RNG to seed so drift is deterministic from the injection point
        self._rng = np.random.default_rng(self.seed)

    def clear_drift(self) -> None:
        """Deactivate drift injection."""
        self._drift_mode = None

    # ------------------------------------------------------------------
    # Streaming
    # ------------------------------------------------------------------

    def stream(
        self,
        drift_at_s: Optional[float] = None,
        drift_mode: DriftMode = "electrode_shift",
    ) -> Iterator[WindowEnvelope]:
        """
        Yield WindowEnvelopes at 1× real-time pace (wall-clock sleep).

        Parameters
        ----------
        drift_at_s : seconds of elapsed replay time before drift is injected.
                     None → no automatic drift (can still be triggered via
                     apply_drift() from another thread).
        drift_mode : which drift type to inject at *drift_at_s*.

        Yields
        ------
        WindowEnvelope TypedDict matching the CLAUDE.md contract.
        """
        assert self._windows is not None, "Session not loaded"
        session = self._session
        assert session is not None

        fs = session.fs
        stride_s = STRIDE_MS / 1000.0   # real-time sleep per window

        t0_wall = time.monotonic()
        t0_ms   = int(time.time() * 1000)

        for i, batch in enumerate(self._windows):
            # Elapsed replay time
            elapsed_s = i * stride_s

            # Auto-inject drift when threshold is crossed
            if (
                drift_at_s is not None
                and self._drift_mode is None
                and elapsed_s >= drift_at_s
            ):
                self.apply_drift(drift_mode)

            # Apply drift (if active)
            window = batch.window
            if self._drift_mode is not None:
                if self._drift_mode == "electrode_shift":
                    # Severe, deterministic whole-array electrode-cap remap,
                    # see apply_electrode_shift_stress's module comment.
                    window = apply_electrode_shift_stress(window, fs)
                else:
                    drift_elapsed = elapsed_s - (drift_at_s or 0.0)
                    window = inject_drift(
                        window,
                        mode=self._drift_mode,
                        rng=self._rng,
                        elapsed_s=max(0.0, drift_elapsed),
                        fs=fs,
                    )

            # Build spectrogram
            png_b64 = window_to_png_b64(window, fs, self._ref_db)

            # Resolve label
            true_gesture: Optional[str] = batch.label_name if batch.label_name else None

            # Recompute RMS / zero-crossing features from the (possibly
            # drift-injected) window, so features reflect any active drift
            # the same way the spectrogram does, rather than always
            # reporting the original clean batch.rms / batch.zero_crossings.
            if self._drift_mode is not None:
                rms_arr = compute_rms(window)
                zc_arr = compute_zero_crossings(window)
            else:
                rms_arr = batch.rms
                zc_arr = batch.zero_crossings

            # Pad/truncate features to exactly 12 elements
            rms_list = rms_arr.tolist()[:N_CHANNELS]
            zc_list  = zc_arr.tolist()[:N_CHANNELS]
            rms_list += [0.0] * (N_CHANNELS - len(rms_list))
            zc_list  += [0]   * (N_CHANNELS - len(zc_list))

            envelope: WindowEnvelope = {
                "t_ms":                 t0_ms + int(elapsed_s * 1000),
                "subject":              self.subject,
                "true_gesture":         true_gesture,
                "spectrogram_png_b64":  png_b64,
                "features": {
                    "rms":              rms_list,
                    "zero_crossings":   zc_list,
                },
            }

            yield envelope

            # Sleep to maintain real-time pace
            next_wall = t0_wall + (i + 1) * stride_s
            sleep_s = next_wall - time.monotonic()
            if sleep_s > 0:
                time.sleep(sleep_s)
