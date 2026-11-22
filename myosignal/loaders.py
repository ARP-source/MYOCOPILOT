"""
myosignal.loaders, load raw sEMG from NinaPro DB2 (.mat) and Meta emg2pose (.h5/.hdf5).

Both loaders introspect the file structure at runtime rather than hard-coding key names.

NinaPro DB2
-----------
- 12 channels, 2000 Hz
- Keys vary by software version (e.g. 'emg'/'EMG', 'stimulus'/'restimulus',
  'repetition'/'rerepetition').  All are discovered by fuzzy key matching.
- Gesture index → POSE_SET name via NINAPRO_LABEL_MAP (NinaPro DB2 Ex1 defaults).

emg2pose mini
-------------
- 16 channels, 2000 Hz
- HDF5 structure is introspected: the first dataset with shape (N, 16) or (16, N)
  is treated as EMG; labels and reps are sought under common group/dataset names.
- An optional metadata.csv (same directory, same stem) may supply user/session/split.
"""

from __future__ import annotations

import csv
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Contract constant, all sources are normalised to this channel count.
# ---------------------------------------------------------------------------
N_CHANNELS: int = 12

# ---------------------------------------------------------------------------
# NinaPro DB2 Exercise-1 gesture-index → POSE_SET name.
# Indices follow the official NinaPro DB2 protocol paper.
# Override by passing a custom label_map to load_ninapro().
# ---------------------------------------------------------------------------
NINAPRO_LABEL_MAP: dict[int, str] = {
    0: "neutral",        # rest
    1: "index_point",    # index finger extension
    2: "index_point",    # middle finger extension (treat as index_point)
    3: "index_point",    # ring finger extension
    4: "index_point",    # little finger extension
    5: "neutral",        # thumb adduction
    6: "fist",           # thumb abduction / power grasp
    7: "pinch",          # lateral pinch
    8: "tripod",         # tripod / 3-finger pinch
    9: "pinch",          # precision pinch
    10: "fist",          # medium wrap / power grasp
    11: "fist",          # ring grasp
    12: "fist",          # large diameter grasp
    13: "tripod",        # precision disk
    14: "wrist_flex",    # wrist flexion
    15: "wrist_extend",  # wrist extension
    16: "pronation",     # forearm pronation
    17: "supination",    # forearm supination
}


# ---------------------------------------------------------------------------
# Output dataclass
# ---------------------------------------------------------------------------

@dataclass
class RawSession:
    """
    Normalised output from any loader.

    Attributes
    ----------
    emg       : float32 array of shape (n_samples, N_CHANNELS=12).
                Sources with fewer channels are zero-padded; more are truncated.
    labels    : int32 array of shape (n_samples,).
                Integer gesture index (source-specific).
    label_names : str array of shape (n_samples,) or empty.
                POSE_SET string per sample ('' where mapping is unknown).
    reps      : int32 array of shape (n_samples,).
                Repetition number per sample (1-indexed; 0 = rest/unlabelled).
    fs        : sampling rate in Hz.
    subject   : string identifier.
    source    : 'ninapro' | 'emg2pose' | 'synthetic'.
    metadata  : optional dict of extra info from CSV / HDF5 attributes.
    """

    emg: np.ndarray
    labels: np.ndarray
    label_names: np.ndarray
    reps: np.ndarray
    fs: int
    subject: str
    source: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_key(available: list[str], candidates: list[str]) -> Optional[str]:
    """Return the first candidate (case-insensitive) found in *available*."""
    lower_map = {k.lower(): k for k in available}
    for c in candidates:
        found = lower_map.get(c.lower())
        if found is not None:
            return found
    return None


def _normalise_channels(emg: np.ndarray) -> np.ndarray:
    """Truncate or zero-pad the channel axis to exactly N_CHANNELS."""
    emg = np.atleast_2d(emg)
    if emg.shape[0] < emg.shape[1]:
        # shape might be (channels, samples), transpose
        if emg.shape[0] in (12, 16, 8, 4):
            emg = emg.T
    n = emg.shape[1]
    if n == N_CHANNELS:
        return emg.astype(np.float32)
    elif n > N_CHANNELS:
        return emg[:, :N_CHANNELS].astype(np.float32)
    else:
        pad = np.zeros((emg.shape[0], N_CHANNELS - n), dtype=np.float32)
        return np.concatenate([emg.astype(np.float32), pad], axis=1)


def _apply_label_map(labels: np.ndarray, label_map: dict[int, str]) -> np.ndarray:
    """Vectorised integer → POSE_SET string lookup."""
    out = np.empty(len(labels), dtype=object)
    for i, lbl in enumerate(labels):
        out[i] = label_map.get(int(lbl), "")
    return out.astype(str)


# ---------------------------------------------------------------------------
# NinaPro loader
# ---------------------------------------------------------------------------

def load_ninapro(
    mat_path: Path,
    subject: str = "",
    label_map: Optional[dict[int, str]] = None,
    fs: int = 2000,
) -> RawSession:
    """
    Load a NinaPro DB2 .mat exercise file.

    Keys are discovered at runtime, no assumptions about variable names.

    Parameters
    ----------
    mat_path   : path to S<n>_E<n>_A1.mat (or any NinaPro .mat variant).
    subject    : identifier string; defaults to the file stem.
    label_map  : integer-label → POSE_SET name; defaults to NINAPRO_LABEL_MAP.
    fs         : nominal sampling rate (NinaPro DB2 = 2000 Hz); overridden if
                 'frequency' / 'freq' key is present in the .mat file.

    Returns
    -------
    RawSession with 12-channel EMG, integer labels, label_names, and reps.
    """
    import scipy.io

    mat_path = Path(mat_path)
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    public_keys = [k for k in mat.keys() if not k.startswith("_")]

    # --- EMG ---
    emg_key = _find_key(public_keys, ["emg", "EMG", "data", "signal", "eeg"])
    if emg_key is None:
        raise KeyError(
            f"Cannot find an EMG array in {mat_path.name}. "
            f"Available keys: {public_keys}"
        )
    emg_raw = np.array(mat[emg_key], dtype=np.float32)
    emg = _normalise_channels(emg_raw)
    n_samples = emg.shape[0]

    # --- Labels ---
    lbl_key = _find_key(public_keys, ["restimulus", "stimulus", "label", "labels",
                                       "class", "gesture", "movement"])
    if lbl_key is not None:
        labels_raw = np.array(mat[lbl_key], dtype=np.int32).ravel()
    else:
        labels_raw = np.zeros(n_samples, dtype=np.int32)

    # --- Repetitions ---
    rep_key = _find_key(public_keys, ["rerepetition", "repetition", "rep", "reps",
                                       "trial", "trials"])
    if rep_key is not None:
        reps_raw = np.array(mat[rep_key], dtype=np.int32).ravel()
    else:
        reps_raw = np.ones(n_samples, dtype=np.int32)

    # --- Sampling rate override ---
    freq_key = _find_key(public_keys, ["frequency", "freq", "fs", "srate",
                                        "samplingfrequency"])
    if freq_key is not None:
        val = mat[freq_key]
        try:
            fs = int(np.atleast_1d(val).ravel()[0])
        except Exception:
            pass

    # Align lengths
    min_len = min(n_samples, len(labels_raw), len(reps_raw))
    emg = emg[:min_len]
    labels = labels_raw[:min_len]
    reps = reps_raw[:min_len]

    lmap = label_map if label_map is not None else NINAPRO_LABEL_MAP
    label_names = _apply_label_map(labels, lmap)

    return RawSession(
        emg=emg,
        labels=labels,
        label_names=label_names,
        reps=reps,
        fs=fs,
        subject=subject or mat_path.stem,
        source="ninapro",
    )


# ---------------------------------------------------------------------------
# emg2pose loader
# ---------------------------------------------------------------------------

def load_emg2pose(
    h5_path: Path,
    subject: str = "",
    metadata_csv: Optional[Path] = None,
    fs: int = 2000,
) -> RawSession:
    """
    Load a Meta emg2pose mini HDF5 file (16 channels, 2 kHz).

    The HDF5 structure is introspected at runtime.  A companion metadata.csv
    (same directory, same stem by default) may contain user/session/split columns.

    Parameters
    ----------
    h5_path      : path to the .h5 / .hdf5 file.
    subject      : identifier string; inferred from metadata.csv if omitted.
    metadata_csv : explicit path to the companion CSV; if None, searched
                   automatically as <h5_path.stem>.csv in the same directory.
    fs           : nominal sampling rate (emg2pose = 2000 Hz); overridden by
                   HDF5 attributes if present.

    Returns
    -------
    RawSession with 12-channel EMG (first 12 of 16), integer labels, and reps.
    """
    import h5py

    h5_path = Path(h5_path)

    # --- Auto-discover metadata CSV ---
    if metadata_csv is None:
        candidates = [
            h5_path.with_suffix(".csv"),
            h5_path.parent / "metadata.csv",
            h5_path.parent / (h5_path.stem + "_metadata.csv"),
        ]
        for c in candidates:
            if c.exists():
                metadata_csv = c
                break

    meta: dict = {}
    if metadata_csv is not None and Path(metadata_csv).exists():
        with open(metadata_csv, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if rows:
            # Use first row for session-level info
            row = rows[0]
            meta = {k.lower(): v for k, v in row.items()}
            if not subject:
                subject = meta.get("user", meta.get("subject", ""))

    emg_array: Optional[np.ndarray] = None
    labels_array: Optional[np.ndarray] = None
    reps_array: Optional[np.ndarray] = None

    def _walk_for_emg(name, obj):
        nonlocal emg_array, fs
        if not isinstance(obj, h5py.Dataset):
            return
        shape = obj.shape
        if len(shape) != 2:
            return
        # Match (N, 16) or (16, N), emg2pose has 16 channels
        if 16 in shape and max(shape) > 16:
            arr = obj[:]
            if arr.shape[1] != 16:
                arr = arr.T
            if emg_array is None or arr.shape[0] > emg_array.shape[0]:
                emg_array = arr.astype(np.float32)
            # Check for fs attribute
            if "fs" in obj.attrs:
                fs = int(obj.attrs["fs"])
            elif "sampling_rate" in obj.attrs:
                fs = int(obj.attrs["sampling_rate"])

    def _walk_for_labels(name, obj):
        nonlocal labels_array
        if not isinstance(obj, h5py.Dataset):
            return
        lname = name.split("/")[-1].lower()
        if lname in ("label", "labels", "gesture", "stimulus", "class",
                     "movement", "annotation"):
            if labels_array is None:
                labels_array = obj[:].ravel().astype(np.int32)

    def _walk_for_reps(name, obj):
        nonlocal reps_array
        if not isinstance(obj, h5py.Dataset):
            return
        rname = name.split("/")[-1].lower()
        if rname in ("rep", "reps", "repetition", "repetitions", "trial", "trials"):
            if reps_array is None:
                reps_array = obj[:].ravel().astype(np.int32)

    with h5py.File(h5_path, "r") as f:
        # Check for fs in root attributes
        for attr in ("fs", "sampling_rate", "frequency"):
            if attr in f.attrs:
                fs = int(f.attrs[attr])
                break
        f.visititems(_walk_for_emg)
        f.visititems(_walk_for_labels)
        f.visititems(_walk_for_reps)

    if emg_array is None:
        # Fallback: open again and grab any 2-D dataset with >1000 rows
        with h5py.File(h5_path, "r") as f:
            def _fallback(name, obj):
                nonlocal emg_array
                if isinstance(obj, h5py.Dataset) and len(obj.shape) == 2:
                    r, c = sorted(obj.shape)
                    if r > 1000 and emg_array is None:
                        arr = obj[:]
                        if arr.shape[0] < arr.shape[1]:
                            arr = arr.T
                        emg_array = arr.astype(np.float32)
            f.visititems(_fallback)

    if emg_array is None:
        raise ValueError(
            f"Could not locate an EMG dataset in {h5_path.name}.  "
            "Expected a 2-D array with 16 columns (or rows)."
        )

    emg = _normalise_channels(emg_array)
    n_samples = emg.shape[0]

    labels = labels_array[:n_samples] if labels_array is not None else np.zeros(n_samples, dtype=np.int32)
    reps   = reps_array[:n_samples]   if reps_array   is not None else np.ones(n_samples,  dtype=np.int32)
    labels = labels[:n_samples]
    reps   = reps[:n_samples]

    # emg2pose uses continuous joint angles, not discrete gesture classes.
    # Map everything to "neutral" until a richer mapping is provided.
    label_names = np.array(["neutral"] * n_samples, dtype=str)

    return RawSession(
        emg=emg,
        labels=labels,
        label_names=label_names,
        reps=reps,
        fs=fs,
        subject=subject or h5_path.stem,
        source="emg2pose",
        metadata=meta,
    )


# ---------------------------------------------------------------------------
# Synthetic session (for tests / demo when no data file is present)
# ---------------------------------------------------------------------------
#
# NOT a clinical or biomechanical simulation. These are deliberately
# engineered "structured EMG-like synthetic signals": each POSE_SET gesture
# (other than 'neutral') gets a fixed, overlapping-but-distinct 12-channel
# relative activation "signature" below, so per-channel RMS and zero-crossing
# features can actually separate gestures, instead of every gesture being
# channel-identical IID noise that differs only by a random per-burst
# amplitude scale (the previous behavior, which gave clean synthetic sessions
# weak gesture separability regardless of classifier quality).
#
# Channel indices are 0-based, matching the project's 12-channel convention
# (myosignal.loaders.N_CHANNELS / WindowFeatures.rms / .zero_crossings).
_SYNTHETIC_GESTURE_SIGNATURES: dict[str, list[float]] = {
    # broad high activation across 0-7, boosted over the central "flexor-side" 2-5
    "fist":         [0.70, 0.70, 1.00, 1.00, 1.00, 1.00, 0.70, 0.70, 0.15, 0.15, 0.15, 0.15],
    # high in 0-2 and 7-8, low in the middle 3-6, moderate in 9-11
    "index_point":  [0.90, 0.90, 0.90, 0.15, 0.15, 0.15, 0.15, 0.90, 0.90, 0.35, 0.35, 0.35],
    # concentrated in 3-5 and 8-9
    "pinch":        [0.15, 0.15, 0.15, 0.90, 0.90, 0.90, 0.15, 0.15, 0.90, 0.90, 0.15, 0.15],
    # broader overlap across 2-6 and 8, but channel 6 (and low ch7, unlike index_point) sets it apart from pinch
    "tripod":       [0.15, 0.15, 0.80, 0.80, 0.80, 0.80, 0.80, 0.15, 0.75, 0.15, 0.15, 0.15],
    # strong distal/flexor pattern across 6-9
    "wrist_flex":   [0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.90, 0.90, 0.90, 0.90, 0.15, 0.15],
    # complementary extensor pattern across 9-11 plus 1-2
    "wrist_extend": [0.15, 0.80, 0.80, 0.15, 0.15, 0.15, 0.15, 0.15, 0.15, 0.90, 0.90, 0.90],
    # asymmetric single-channel emphasis: 0, 4, 7, 10
    "pronation":    [0.85, 0.15, 0.15, 0.15, 0.85, 0.15, 0.15, 0.85, 0.15, 0.15, 0.85, 0.15],
    # complementary asymmetric emphasis: 2, 5, 8, 11
    "supination":   [0.15, 0.15, 0.85, 0.15, 0.15, 0.85, 0.15, 0.15, 0.85, 0.15, 0.15, 0.85],
}

_SYNTHETIC_BASELINE_NOISE_SCALE = 0.04  # resting/'neutral' noise amplitude
_SYNTHETIC_ACTIVE_AMPLITUDE = 0.55      # peak active-burst amplitude scale
_SYNTHETIC_BURST_S = 1.0                # active-gesture burst duration, seconds
# Must comfortably exceed myosignal.windows.WINDOW_MS (250 ms): a rest period
# shorter than one feature window guarantees every 'neutral' window straddles
# the adjacent active burst and is never actually clean.
_SYNTHETIC_REST_S = 0.5                 # rest ('neutral') between bursts, seconds
_SYNTHETIC_SUBJECT_GAIN_RANGE = (0.85, 1.15)  # per-subject, per-channel fixed gain
_SYNTHETIC_REP_JITTER_RANGE = (0.85, 1.15)    # per-repetition amplitude jitter
_SYNTHETIC_OSC_FREQ_RANGE_HZ = (8.0, 22.0)    # mid-frequency EMG-like component
_SYNTHETIC_BOOTSTRAP_TOPUP_REPS = 2      # extra repetitions of the *first* scheduled
                                         # gesture, immediately after the one diversity
                                         # lap below, so it (and 'neutral', which recurs
                                         # between every repetition) reach a stable
                                         # few-shot repetition count quickly rather than
                                         # waiting for further full laps


def _synthetic_seed(subject: str, seed: int) -> int:
    """
    Combine *subject* and *seed* into one deterministic RNG seed.

    The same (subject, seed) pair always reproduces byte-identical output
    (satisfying make_synthetic_session's determinism contract), while
    different subjects get distinct, but still fully deterministic,
    per-subject channel gains and noise draws. zlib.crc32 is used rather
    than Python's built-in hash() because str hashing is randomized per
    process unless PYTHONHASHSEED is fixed, which would break reproducibility.
    """
    subject_component = zlib.crc32(subject.encode("utf-8"))
    return (seed * 1_000_003 + subject_component) % (2**32 - 1)


def make_synthetic_session(
    n_seconds: float = 72.0,
    fs: int = 2000,
    subject: str = "synthetic",
    seed: int = 0,
) -> RawSession:
    """
    Generate a reproducible, structured EMG-like synthetic sEMG session for
    tests and demos.

    NOT a clinical or biomechanical simulation, see the module-level
    comment above _SYNTHETIC_GESTURE_SIGNATURES. Each active POSE_SET
    gesture is given a fixed, deliberately overlapping-but-distinct
    12-channel relative activation signature (e.g. 'fist' activates channels
    0-7 broadly; 'pinch' concentrates on channels 3-5 and 8-9), so RMS and
    zero-crossing features can actually separate gestures instead of every
    gesture looking like channel-identical noise with only a random overall
    amplitude scale.

    Each active burst layers, per channel:
      - filtered broadband noise, scaled by that channel's signature weight;
      - a mid-frequency (8-22 Hz) oscillatory component (a coarse stand-in
        for motor-unit firing-rate modulation), giving a mixture of
        low/mid-frequency structure plus noise rather than noise alone;
      - a fixed per-subject, per-channel gain (a stand-in for stable
        electrode-placement/anatomy differences between subjects);
      - a per-repetition amplitude and phase jitter, so repeated instances
        of the same gesture are similar but not identical.

    'neutral' is low-amplitude filtered noise only, both during its own
    rest periods between bursts and as the resting baseline under every
    active burst.

    Burst schedule: one round-robin "diversity lap" through all 8 active
    POSE_SET gestures (in POSE_SET order, each rep separated by a 'neutral'
    rest), so a consumer that bootstraps a feature normalization/scaling
    step from the first several clean, labeled windows sees every gesture
    type at least once, instead of calibrating against just the first one
    or two gesture types it happens to see repeated. That's followed by
    _SYNTHETIC_BOOTSTRAP_TOPUP_REPS extra repetitions of just the first
    gesture (quickly giving it, and 'neutral', a stable few-shot count),
    then an indefinite round-robin cycle through all 8 for the rest of the
    session's duration, for ongoing session dynamics (drift injection,
    recalibration against an already-known gesture).

    Parameters
    ----------
    n_seconds : total duration in seconds.
    fs        : sampling rate in Hz.
    subject   : subject identifier. Also seeds a fixed per-subject,
                per-channel gain (see _synthetic_seed), so different subjects
                produce different, but each internally deterministic and
                reproducible, signals for the same *seed*.
    seed      : RNG seed for reproducibility. The same (subject, seed) pair
                always yields a numerically identical session.
    """
    from myosignal.types import POSE_SET
    from scipy.signal import butter, filtfilt

    rng = np.random.default_rng(_synthetic_seed(subject, seed))
    n = int(n_seconds * fs)

    # --- Per-subject fixed per-channel gain (anatomy/electrode-placement stand-in) ---
    gain_lo, gain_hi = _SYNTHETIC_SUBJECT_GAIN_RANGE
    subject_gain = rng.uniform(gain_lo, gain_hi, size=N_CHANNELS).astype(np.float32)

    # --- Baseline: low-amplitude band-limited ('neutral') noise under the whole session ---
    b, a = butter(4, [20 / (fs / 2), 450 / (fs / 2)], btype="bandpass")
    emg = rng.standard_normal((n, N_CHANNELS)).astype(np.float32) * _SYNTHETIC_BASELINE_NOISE_SCALE
    for ch in range(N_CHANNELS):
        emg[:, ch] = filtfilt(b, a, emg[:, ch]).astype(np.float32)

    labels = np.zeros(n, dtype=np.int32)
    reps = np.zeros(n, dtype=np.int32)
    label_names = np.array(["neutral"] * n, dtype=object)

    gesture_cycle = POSE_SET[1:]  # 8 non-neutral gestures, in POSE_SET order
    burst_len = int(_SYNTHETIC_BURST_S * fs)
    rest_len = int(_SYNTHETIC_REST_S * fs)
    jitter_lo, jitter_hi = _SYNTHETIC_REP_JITTER_RANGE
    freq_lo, freq_hi = _SYNTHETIC_OSC_FREQ_RANGE_HZ

    # Schedule: one diversity lap (all 8, interleaved) + a quick top-up of
    # the first gesture, then an indefinite round-robin cycle for the rest
    # of the session's duration, see the docstring's "Burst schedule"
    # section. The indefinite cycle (rather than a fixed number of extra
    # laps) means the ratio of the bootstrap-only prefix to the whole
    # session shrinks as n_seconds grows, instead of a fixed-length
    # schedule leaving the back half of a long session as unlabeled
    # baseline noise.
    schedule_prefix: list[str] = list(gesture_cycle) + [gesture_cycle[0]] * _SYNTHETIC_BOOTSTRAP_TOPUP_REPS

    idx = 0
    rep_num = 1
    burst_i = 0
    while idx + burst_len <= n:
        if burst_i < len(schedule_prefix):
            gesture = schedule_prefix[burst_i]
        else:
            gesture = gesture_cycle[(burst_i - len(schedule_prefix)) % len(gesture_cycle)]
        burst_i += 1
        signature = _SYNTHETIC_GESTURE_SIGNATURES[gesture]

        rep_scale = rng.uniform(jitter_lo, jitter_hi)   # repetition-level amplitude jitter
        phase = rng.uniform(0.0, 2 * np.pi)             # repetition-level phase jitter
        osc_freq = rng.uniform(freq_lo, freq_hi)        # mid-frequency EMG-like component

        t = np.arange(burst_len, dtype=np.float32) / fs
        ramp = np.sin(np.pi * np.clip(t / (burst_len / fs), 0.0, 1.0)).astype(np.float32)
        osc = np.sin(2 * np.pi * osc_freq * t + phase).astype(np.float32)
        noise = rng.standard_normal((burst_len, N_CHANNELS)).astype(np.float32)

        for ch in range(N_CHANNELS):
            channel_gain = (
                signature[ch] * subject_gain[ch] * rep_scale * _SYNTHETIC_ACTIVE_AMPLITUDE
            )
            emg[idx:idx + burst_len, ch] += (noise[:, ch] * 0.65 + osc * 0.35) * channel_gain * ramp

        labels[idx:idx + burst_len] = POSE_SET.index(gesture)
        label_names[idx:idx + burst_len] = gesture
        reps[idx:idx + burst_len] = rep_num

        idx += burst_len + rest_len
        rep_num += 1

    return RawSession(
        emg=emg,
        labels=labels,
        label_names=label_names.astype(str),
        reps=reps,
        fs=fs,
        subject=subject,
        source="synthetic",
    )
