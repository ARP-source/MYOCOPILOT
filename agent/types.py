"""
Shared types for the agent package.
"""

from __future__ import annotations
from typing import TypedDict


class ClassifierResponse(TypedDict):
    """
    What the model layer must return for every window.

    gesture        : predicted gesture label (must be in POSE_SET)
    confidence     : model confidence in [0.0, 1.0]
    drift_suspected: True when the model observes anomalous signal characteristics
    notes          : free-text reasoning or explanation from the LLM
    """

    gesture: str
    confidence: float
    drift_suspected: bool
    notes: str


class RecalibrationResult(TypedDict):
    """
    Outcome of a completed recalibration round.

    gesture   : the gesture that was recalibrated
    acc_before: rolling accuracy immediately before recalibration
    acc_after : rolling accuracy immediately after recalibration
    """

    gesture: str
    acc_before: float
    acc_after: float
