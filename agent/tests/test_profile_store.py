"""
Tests for agent.profile_store, load/save round-trip and example eviction.
"""

from __future__ import annotations

from agent import profile_store


class TestLoadProfile:
    def test_missing_subject_returns_empty_structure(self, isolated_profiles_dir):
        profile = profile_store.load_profile("nobody")
        assert profile == {"subject": "nobody", "gestures": {}}

    def test_round_trips_through_save(self, isolated_profiles_dir):
        profile = {"subject": "s1", "gestures": {"fist": {"examples": ["aaa"], "updated_at_ms": 1}}}
        profile_store.save_profile("s1", profile)
        assert profile_store.load_profile("s1") == profile


class TestAppendExamples:
    def test_creates_gesture_entry(self, isolated_profiles_dir):
        profile_store.append_examples("s2", "fist", ["img1", "img2"])
        profile = profile_store.load_profile("s2")
        assert profile["gestures"]["fist"]["examples"] == ["img1", "img2"]
        assert profile["gestures"]["fist"]["updated_at_ms"] > 0

    def test_accumulates_across_calls(self, isolated_profiles_dir):
        profile_store.append_examples("s3", "pinch", ["a"])
        profile_store.append_examples("s3", "pinch", ["b", "c"])
        profile = profile_store.load_profile("s3")
        assert profile["gestures"]["pinch"]["examples"] == ["a", "b", "c"]

    def test_evicts_oldest_past_max(self, isolated_profiles_dir):
        limit = profile_store.MAX_EXAMPLES_PER_GESTURE
        profile_store.append_examples("s4", "tripod", [f"ex{i}" for i in range(limit)])
        profile_store.append_examples("s4", "tripod", ["newest"])

        examples = profile_store.load_profile("s4")["gestures"]["tripod"]["examples"]

        assert len(examples) == limit
        assert examples[-1] == "newest"
        assert examples[0] == "ex1"  # ex0 evicted
