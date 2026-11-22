"""
eval._common, shared session loading/segmentation helpers for the eval/ package.

Every script under eval/ (run.py, baselines.py) needs the same thing: find up
to 3 subjects (real NinaPro/emg2pose files under data/, falling back to
deterministic synthetic sessions), preprocess them, and segment them into
windows.  That logic lives here once instead of being duplicated.

This intentionally talks to myosignal.loaders / myosignal.preprocess /
myosignal.windows directly rather than going through myosignal.stream() /
SessionPlayer.stream().  Those replay at 1x real-time wall-clock pace (by
design, for the live dashboard) and render a spectrogram PNG for every
window, both are wasted work for an offline batch evaluation that only
needs the RMS/zero-crossing features and ground-truth labels. No file
outside eval/ is modified to make this possible; everything here calls
already-public myosignal functions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, TypedDict

from myosignal.loaders import RawSession, load_ninapro, load_emg2pose, make_synthetic_session
from myosignal.preprocess import preprocess
from myosignal.windows import WindowBatch, segment, STRIDE_MS, WINDOW_MS

DATA_DIR = Path("data")
N_SUBJECTS = 3
SYNTHETIC_DATA_LABEL = "synthetic_demo_data"
REAL_DATA_LABEL = "real"


class SubjectDescriptor(TypedDict):
    subject_id: str
    source: str            # "ninapro" | "emg2pose" | "synthetic"
    path: Optional[Path]


def discover_subjects(n: int = N_SUBJECTS, data_dir: Path = DATA_DIR) -> list[SubjectDescriptor]:
    """
    Return up to *n* subject descriptors.

    Real NinaPro (.mat) and emg2pose (.h5/.hdf5) files under *data_dir* are
    preferred, sorted by filename for determinism. Any remaining slots (all
    of them, if *data_dir* has no usable files) are filled with deterministic
    synthetic subjects, clearly labeled via data_source_label().
    """
    real: list[SubjectDescriptor] = []
    if data_dir.exists():
        for p in sorted(data_dir.glob("*.mat")):
            real.append({"subject_id": p.stem, "source": "ninapro", "path": p})
        for p in sorted(list(data_dir.glob("*.h5")) + list(data_dir.glob("*.hdf5"))):
            real.append({"subject_id": p.stem, "source": "emg2pose", "path": p})

    subjects: list[SubjectDescriptor] = real[:n]
    synth_idx = 1
    while len(subjects) < n:
        subjects.append(
            {"subject_id": f"synthetic_subject_{synth_idx}", "source": "synthetic", "path": None}
        )
        synth_idx += 1
    return subjects


def data_source_label(descriptor: SubjectDescriptor) -> str:
    """Return 'real' for NinaPro/emg2pose files, 'synthetic_demo_data' otherwise."""
    return REAL_DATA_LABEL if descriptor["source"] in ("ninapro", "emg2pose") else SYNTHETIC_DATA_LABEL


def load_raw_session(descriptor: SubjectDescriptor, seed: int) -> RawSession:
    """Load the RawSession for *descriptor*, using *seed* only for synthetic subjects."""
    if descriptor["source"] == "ninapro":
        assert descriptor["path"] is not None
        return load_ninapro(descriptor["path"], subject=descriptor["subject_id"])
    if descriptor["source"] == "emg2pose":
        assert descriptor["path"] is not None
        return load_emg2pose(descriptor["path"], subject=descriptor["subject_id"])
    return make_synthetic_session(subject=descriptor["subject_id"], seed=seed)


def preprocess_and_segment(raw: RawSession, baseline_s: float = 30.0) -> list[WindowBatch]:
    """Bandpass-filter, z-score, and segment *raw* into WindowBatches (no notch, matching SessionPlayer defaults)."""
    emg_clean, _, _ = preprocess(raw.emg, raw.fs, notch=False, baseline_s=baseline_s)
    return segment(
        emg_clean,
        labels=raw.labels,
        label_names=raw.label_names,
        reps=raw.reps,
        fs=raw.fs,
    )


def load_and_segment(descriptor: SubjectDescriptor, seed: int) -> tuple[RawSession, list[WindowBatch]]:
    """Convenience: load + preprocess + segment in one call."""
    raw = load_raw_session(descriptor, seed=seed)
    batches = preprocess_and_segment(raw)
    return raw, batches
