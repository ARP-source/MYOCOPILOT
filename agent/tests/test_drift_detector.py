"""
Tests for agent.drift_detector — rolling accuracy, confusion tracking,
confidence-drop detection, and the debounced check_drift() trigger.
"""

from __future__ import annotations

from agent.drift_detector import DriftDetector, check_drift, DEBOUNCE_S


def _pred(gesture: str, confidence: float = 0.9) -> dict:
    return {"gesture": gesture, "confidence": confidence, "drift_suspected": False, "notes": ""}


class TestRollingAccuracy:
    def test_defaults_to_1_when_no_labeled_windows(self):
        d = DriftDetector()
        assert d.rolling_accuracy() == 1.0

    def test_tracks_correct_vs_incorrect(self):
        d = DriftDetector(window_size=4)
        d.update("fist", _pred("fist"))
        d.update("fist", _pred("pinch"))
        d.update("fist", _pred("fist"))
        d.update("fist", _pred("fist"))
        assert d.rolling_accuracy() == 0.75

    def test_window_is_bounded(self):
        d = DriftDetector(window_size=3)
        for _ in range(3):
            d.update("fist", _pred("pinch"))  # all wrong, fills window
        d.update("fist", _pred("fist"))  # correct — should push out one wrong entry
        assert d.rolling_accuracy() == 1 / 3


class TestDriftNotPrematurelyTriggered:
    def test_single_early_miss_does_not_trigger(self):
        """Regression: a lone 0/1 miss must not read as a 0% rolling accuracy drift."""
        d = DriftDetector(window_size=20, accuracy_threshold=0.60)
        d.update("fist", _pred("pinch"))
        assert check_drift(d) is None

    def test_few_misses_below_window_size_do_not_trigger(self):
        d = DriftDetector(window_size=20, accuracy_threshold=0.60)
        for _ in range(5):
            d.update("fist", _pred("pinch"))
        assert check_drift(d) is None


class TestDriftTriggersOnLowAccuracy:
    def test_triggers_and_returns_dominant_pair(self):
        d = DriftDetector(window_size=20, accuracy_threshold=0.60, confusion_threshold=0.25)
        for _ in range(20):
            d.update("fist", _pred("pinch"))

        pair = check_drift(d)

        assert pair == ("fist", "pinch")

    def test_debounced_within_90s(self):
        d = DriftDetector(window_size=20, accuracy_threshold=0.60)
        for _ in range(20):
            d.update("fist", _pred("pinch"))

        first = check_drift(d)
        second = check_drift(d)

        assert first == ("fist", "pinch")
        assert second is None  # debounced

    def test_debounce_window_matches_spec(self):
        assert DEBOUNCE_S == 90.0


class TestConfidenceDrop:
    def test_low_confidence_alone_triggers_even_without_labels(self):
        d = DriftDetector(window_size=20, accuracy_threshold=0.60)
        # Establish a long high-confidence baseline (unlabeled windows), like
        # a clean multi-minute session before drift starts.
        for _ in range(200):
            d.update(None, _pred("fist", confidence=0.95))
        # Then a sharp, sustained confidence drop over exactly one window.
        for _ in range(20):
            d.update(None, _pred("fist", confidence=0.1))

        assert d.confidence_dropped() is True
        # Rolling accuracy stays 1.0 (no labeled data) — only confidence signals drift.
        assert d.rolling_accuracy() == 1.0

    def test_stable_confidence_does_not_trigger(self):
        d = DriftDetector(window_size=20)
        for _ in range(40):
            d.update(None, _pred("fist", confidence=0.9))
        assert d.confidence_dropped() is False


class TestDominantConfusion:
    def test_returns_none_below_threshold(self):
        d = DriftDetector(window_size=20, confusion_threshold=0.5)
        d.update("fist", _pred("pinch"))
        for _ in range(19):
            d.update("fist", _pred("fist"))
        assert d.dominant_confusion() is None

    def test_top_confusion_pair_ignores_threshold(self):
        d = DriftDetector(window_size=20, confusion_threshold=0.9)
        d.update("fist", _pred("pinch"))
        for _ in range(19):
            d.update("fist", _pred("fist"))
        assert d.dominant_confusion() is None
        assert d.top_confusion_pair() == ("fist", "pinch")
