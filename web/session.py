"""
web.session — per-WebSocket-connection state.

Each connected client gets its own Session object that tracks classification
history, the active DriftDetector, and a CostTracker.
"""

from __future__ import annotations

from myosignal.types import POSE_SET
from agent.types import ClassifierResponse
from agent.drift_detector import DriftDetector
from agent.cost_tracker import CostTracker


class Session:
    """
    Container for all mutable state belonging to one WebSocket connection.

    Attributes
    ----------
    subject        : dataset subject being replayed
    history        : list of classification result dicts (append-only)
    accuracy_series: list of {"t_ms": int, "acc": float} snapshots
    confusion      : 9x9 confusion matrix (POSE_SET order), as list[list[int]]
    detector       : active DriftDetector for this session
    cost           : CostTracker accumulating API spend
    playing        : whether the stream is currently running (not paused)
    """

    def __init__(self, subject: str) -> None:
        """
        Initialise a Session for the given subject.

        Parameters
        ----------
        subject:
            Dataset subject identifier.
        """
        self.subject = subject
        self.history: list[dict] = []
        self.accuracy_series: list[dict] = []
        self.confusion: list[list[int]] = [
            [0] * len(POSE_SET) for _ in POSE_SET
        ]
        self.detector = DriftDetector()
        self.cost = CostTracker()
        self.playing = True

    def record_classification(
        self,
        t_ms: int,
        true_gesture: str | None,
        pred_gesture: str,
        confidence: float,
        drift_suspected: bool,
        notes: str,
    ) -> None:
        """
        Append one classification result to history and update the confusion
        matrix and accuracy_series.

        Parameters
        ----------
        t_ms          : window timestamp in ms
        true_gesture  : ground-truth label (may be None)
        pred_gesture  : predicted gesture label
        confidence    : model confidence
        drift_suspected: whether the model flagged drift
        notes         : LLM free-text notes
        """
        pred: ClassifierResponse = {
            "gesture": pred_gesture,
            "confidence": confidence,
            "drift_suspected": drift_suspected,
            "notes": notes,
        }
        self.detector.update(true_gesture, pred)

        if true_gesture in POSE_SET and pred_gesture in POSE_SET:
            self.confusion[POSE_SET.index(true_gesture)][POSE_SET.index(pred_gesture)] += 1

        rolling_accuracy_20 = self.detector.rolling_accuracy()
        self.accuracy_series.append({"t_ms": t_ms, "acc": rolling_accuracy_20})

        self.history.append(
            {
                "t_ms": t_ms,
                "true_gesture": true_gesture,
                "pred": pred,
                "rolling_accuracy_20": rolling_accuracy_20,
                "cumulative_cost_usd": self.cost.cost_usd,
            }
        )

    def confusion_matrix_list(self) -> list[list[int]]:
        """
        Return the confusion matrix as a nested list (9x9).

        Row index = true gesture index in POSE_SET.
        Column index = predicted gesture index in POSE_SET.
        """
        return self.confusion
