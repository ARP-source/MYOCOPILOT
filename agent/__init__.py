"""
agent, classifier, drift detection, conversational recalibration, cost tracking.

Public API:
    classify(envelope) -> ClassifierResponse
    check_drift(history)  -> str | None   (gesture name if drift detected, else None)
    recalibrate(session, gesture) -> RecalibrationResult
"""

import os

from dotenv import load_dotenv

load_dotenv()

if os.environ.get("MOCK") == "1":
    from agent.mock_classifier import classify
else:
    from agent.classifier import classify

from agent.drift_detector import check_drift
from agent.recalibrator import recalibrate
from agent.types import ClassifierResponse

__all__ = ["classify", "check_drift", "recalibrate", "ClassifierResponse"]
