"""
agent.profile_store, load and save per-user gesture profiles.

Each profile is a JSON file at gesture_profiles/<subject>.json containing
few-shot spectrogram examples keyed by gesture name.

Schema:
    {
        "subject": "<str>",
        "gestures": {
            "<gesture_name>": {
                "examples": ["<base64_png>", ...],   // up to MAX_EXAMPLES
                "updated_at_ms": <int>
            },
            ...
        }
    }
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

PROFILES_DIR = Path("gesture_profiles")
MAX_EXAMPLES_PER_GESTURE = 20


def _profile_path(subject: str) -> Path:
    return PROFILES_DIR / f"{subject}.json"


def _empty_profile(subject: str) -> dict:
    return {"subject": subject, "gestures": {}}


def load_profile(subject: str) -> dict:
    """
    Load the gesture profile for *subject* from disk.

    Creates and returns an empty profile structure if no file exists yet.

    Parameters
    ----------
    subject:
        Subject identifier (used as the filename stem).

    Returns
    -------
    Profile dict matching the schema described in the module docstring.
    """
    path = _profile_path(subject)
    if not path.exists():
        return _empty_profile(subject)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_profile(subject: str, profile: dict) -> None:
    """
    Persist *profile* to gesture_profiles/<subject>.json (atomic write).

    Parameters
    ----------
    subject:
        Subject identifier.
    profile:
        Profile dict matching the schema described in the module docstring.
    """
    PROFILES_DIR.mkdir(parents=True, exist_ok=True)
    path = _profile_path(subject)
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(profile, f, indent=2)
    os.replace(tmp_path, path)


def append_examples(subject: str, gesture: str, png_b64_list: list[str]) -> None:
    """
    Add new spectrogram examples for *gesture* to the subject's profile,
    evicting the oldest ones when MAX_EXAMPLES_PER_GESTURE is exceeded.

    Parameters
    ----------
    subject:
        Subject identifier.
    gesture:
        Gesture name (must be in POSE_SET).
    png_b64_list:
        List of base64 PNG strings to append.
    """
    profile = load_profile(subject)
    entry = profile["gestures"].setdefault(
        gesture, {"examples": [], "updated_at_ms": 0}
    )
    entry["examples"].extend(png_b64_list)
    entry["examples"] = entry["examples"][-MAX_EXAMPLES_PER_GESTURE:]
    entry["updated_at_ms"] = int(time.time() * 1000)
    save_profile(subject, profile)
