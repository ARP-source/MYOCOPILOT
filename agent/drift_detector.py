"""
agent.drift_detector, rolling accuracy monitoring and drift heuristics.

Maintains a sliding window of recent (true_gesture, predicted_gesture) pairs
and raises a drift alert when rolling accuracy falls below a threshold or when
a specific confusion pattern dominates.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional

from agent.types import ClassifierResponse

# Confidence drop threshold, in standard deviations below the running baseline.
CONFIDENCE_STDDEV_FACTOR = 2.0

# Minimum seconds between two drift triggers (see check_drift()).
DEBOUNCE_S = 90.0


class DriftDetector:
    """
    Stateful detector that tracks rolling accuracy and per-gesture confusion.

    Parameters
    ----------
    window_size:
        Number of recent predictions to keep for rolling accuracy.
    accuracy_threshold:
        Rolling accuracy below this value triggers drift_suspected.
    confusion_threshold:
        If a single (true, pred) confusion pair exceeds this fraction of the
        window, a targeted recalibration is triggered.
    """

    def __init__(
        self,
        window_size: int = 20,
        accuracy_threshold: float = 0.60,
        confusion_threshold: float = 0.25,
    ) -> None:
        self.window_size = window_size
        self.accuracy_threshold = accuracy_threshold
        self.confusion_threshold = confusion_threshold

        # (true_gesture, pred_gesture) for the last *window_size* labeled windows.
        self._recent: deque[tuple[str, str]] = deque(maxlen=window_size)
        # confidence of the last *window_size* windows (labeled or not).
        self._recent_confidence: deque[float] = deque(maxlen=window_size)

        # Running mean/variance of confidence over every window seen so far
        # (Welford's online algorithm), the "baseline" the rolling window is
        # compared against.
        self._conf_n = 0
        self._conf_mean = 0.0
        self._conf_m2 = 0.0

        self._last_trigger_wall: Optional[float] = None

    def update(self, true_gesture: str | None, pred: ClassifierResponse) -> None:
        """
        Record one prediction.

        Parameters
        ----------
        true_gesture:
            Ground-truth label (may be None during live operation).
        pred:
            ClassifierResponse returned by the classifier.
        """
        confidence = pred["confidence"]
        self._recent_confidence.append(confidence)

        self._conf_n += 1
        delta = confidence - self._conf_mean
        self._conf_mean += delta / self._conf_n
        self._conf_m2 += delta * (confidence - self._conf_mean)

        if true_gesture is not None:
            self._recent.append((true_gesture, pred["gesture"]))

    def rolling_accuracy(self) -> float:
        """
        Return accuracy over the last *window_size* labeled predictions.

        Returns 1.0 if no labeled predictions have been seen yet.
        """
        if not self._recent:
            return 1.0
        correct = sum(1 for t, p in self._recent if t == p)
        return correct / len(self._recent)

    def _confusion_counts(self) -> dict[tuple[str, str], int]:
        counts: dict[tuple[str, str], int] = {}
        for t, p in self._recent:
            if t != p:
                counts[(t, p)] = counts.get((t, p), 0) + 1
        return counts

    def dominant_confusion(self) -> tuple[str, str] | None:
        """
        Return the (true_gesture, confused_with) pair that exceeds
        *confusion_threshold*, or None if no such pair exists.
        """
        counts = self._confusion_counts()
        if not counts:
            return None
        pair, count = max(counts.items(), key=lambda kv: kv[1])
        if count / len(self._recent) >= self.confusion_threshold:
            return pair
        return None

    def top_confusion_pair(self) -> tuple[str, str] | None:
        """
        Return the most frequent (true_gesture, confused_with) pair in the
        current window regardless of *confusion_threshold*, or None if every
        recent prediction was correct.
        """
        counts = self._confusion_counts()
        if not counts:
            return None
        return max(counts.items(), key=lambda kv: kv[1])[0]

    def confidence_dropped(self) -> bool:
        """
        True when the mean confidence of the last *window_size* windows falls
        *CONFIDENCE_STDDEV_FACTOR* standard deviations below the running
        baseline confidence.
        """
        if len(self._recent_confidence) < self.window_size or self._conf_n < 2:
            return False
        baseline_std = (self._conf_m2 / (self._conf_n - 1)) ** 0.5
        if baseline_std == 0.0:
            return False
        window_mean = sum(self._recent_confidence) / len(self._recent_confidence)
        return window_mean < self._conf_mean - CONFIDENCE_STDDEV_FACTOR * baseline_std

    def drift_suspected(self) -> bool:
        """True when accuracy or confidence signals indicate drift (ignoring debounce).

        Accuracy-based drift requires a full window of labeled predictions,
        otherwise a single early misclassification (e.g. 0/1) would trigger
        drift immediately.
        """
        accuracy_drift = (
            len(self._recent) >= self.window_size
            and self.rolling_accuracy() < self.accuracy_threshold
        )
        return accuracy_drift or self.confidence_dropped()

    def debounced(self) -> bool:
        """True if a drift trigger fired within the last DEBOUNCE_S seconds."""
        if self._last_trigger_wall is None:
            return False
        return (time.monotonic() - self._last_trigger_wall) < DEBOUNCE_S

    def mark_triggered(self) -> None:
        """Record that a drift trigger just fired, starting the debounce window."""
        self._last_trigger_wall = time.monotonic()


def check_drift(detector: DriftDetector, feature_detector=None) -> tuple[str, str] | None:
    """
    Convenience wrapper: return the confusion pair to recalibrate when drift
    is detected, or None otherwise.

    Parameters
    ----------
    detector:
        A DriftDetector instance that has been updated with recent predictions.
        Its confusion counts name the gesture to recalibrate, and it holds
        the debounce timer.
    feature_detector:
        Optional agent.feature_drift.FeatureDriftDetector. Once it is fitted,
        its label free alarm decides whether drift is happening, and the
        accuracy and confidence rules in *detector* are not used. On real
        NinaPro DB2 data the accuracy rule was true on about 81% of clean
        windows (eval/drift_results.json), so it is only a fallback while the
        feature detector is still collecting its enrollment windows.

    Returns
    -------
    (target_gesture, confused_with) tuple, or None. With the feature
    detector, None is also returned when it alarms but no labeled confusion
    has been seen, because there is then no gesture to name.
    """
    if detector.debounced():
        return None
    if feature_detector is not None and feature_detector.fitted:
        suspected = feature_detector.alarm
    else:
        suspected = detector.drift_suspected()
    if not suspected:
        return None

    pair = detector.dominant_confusion() or detector.top_confusion_pair()
    if pair is None:
        return None

    detector.mark_triggered()
    return pair
