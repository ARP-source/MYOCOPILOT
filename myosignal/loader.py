"""
myosignal.loader, load raw sEMG data from NinaPro DB2 (.mat) and
Meta emg2pose mini (.hdf5) datasets into a common numpy array format.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np


def load_ninapro_db2(mat_path: Path, subject: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a NinaPro DB2 .mat file and return raw sEMG and label arrays.

    Parameters
    ----------
    mat_path:
        Absolute path to the .mat exercise file (e.g. S1_E1_A1.mat).
    subject:
        Human-readable subject identifier used to tag WindowEnvelopes.

    Returns
    -------
    emg : np.ndarray, shape (N, 12), dtype float32
        Raw sEMG samples at 2000 Hz, 12 channels.
    labels : np.ndarray, shape (N,), dtype int
        Ground-truth gesture index per sample (0 = rest).
    """
    raise NotImplementedError


def load_emg2pose_mini(hdf5_path: Path, subject: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the Meta emg2pose mini HDF5 file and return raw sEMG and label arrays.

    Parameters
    ----------
    hdf5_path:
        Absolute path to the .hdf5 file.
    subject:
        Human-readable subject identifier used to tag WindowEnvelopes.

    Returns
    -------
    emg : np.ndarray, shape (N, 16), dtype float32
        Raw sEMG samples (channel count may differ from NinaPro; windower pads/truncates).
    labels : np.ndarray, shape (N,), dtype int
        Ground-truth gesture index per sample.
    """
    raise NotImplementedError
