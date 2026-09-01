"""
agent.mock_classifier — network-free, deterministic per-subject prototype
classifier, selected via MOCK=1 (see agent/__init__.py's dispatch).

This is NOT the real gesture classifier. agent.classifier.classify() is the
real path: it sends the WindowEnvelope's spectrogram PNG to GLM-5.3-Flash
over OpenRouter and asks a general-purpose multimodal LLM to judge the
visual spectral pattern, optionally comparing it against few-shot reference
images from the subject's gesture_profiles/*.json. This module never makes
a network call and never looks at spectrogram_png_b64 — it classifies
purely from the WindowEnvelope's numeric features (features.rms,
features.zero_crossings), so MOCK=1 replay (the live dashboard demo and the
eval/ harness) never depends on network access or an OpenRouter API key.

Design
------
Per subject (keyed by envelope["subject"]), an in-memory (process-lifetime)
nearest-prototype classifier is built up in two phases:

1. Bootstrap
   The first CLEAN, true-labeled windows are used to (a) freeze a
   per-dimension z-score normalisation (mean/std pooled across the
   bootstrap feature vectors) and (b) enroll a small, deterministic
   reference-example set per gesture. Enrollment only fires a few windows
   into a label run (ENROLL_SETTLE_WINDOWS), never on the first,
   boundary-contaminated window of a run, and at most once per run, up to
   PROTO_CAP examples per gesture — "a small deterministic warm-up
   reference set for every available gesture". The subject stays in
   warm-up (not "ready") until at least MIN_DISTINCT_GESTURES_READY
   distinct gestures have reached MIN_SAMPLES_PER_GESTURE examples; only
   envelope["true_gesture"] ever drives enrollment, never this module's
   own prediction. While warming up, classify() never triggers drift
   (drift_suspected is always False) and communicates the not-ready state
   through the existing `notes` field — no new contract fields.

2. Inference
   For every window (labeled or not), the raw feature vector is
   normalised with the frozen bootstrap statistics and compared against
   every enrolled gesture's stored examples; the nearest example's gesture
   wins. envelope["true_gesture"] is NEVER returned as the prediction —
   only used (above) to decide what to enroll. Confidence comes from the
   margin between the nearest and second-nearest match, relative to the
   population's typical match distance; drift_suspected flags a match
   that is unfamiliar (far from anything ever enrolled) or a confidence
   collapse — not merely "a label happened to be attached".

3. Recalibration
   agent.recalibrator.recalibrate() feeds exactly the 3 confirmed windows
   of the target gesture to enroll_confirmed_samples(), which REPLACES
   that gesture's stored examples outright (a deterministic reset, not a
   blend) and refreshes its audit metadata — see get_profile_snapshot().
"""

from __future__ import annotations

import math
import time
from typing import Optional

from myosignal.types import WindowEnvelope, POSE_SET
from agent.types import ClassifierResponse
from agent.cost_tracker import CostTracker

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------
ZC_NORM_SCALE = 50.0              # rough per-window zero-crossing ceiling, for feature scaling
MIN_SAMPLES_PER_GESTURE = 3        # bootstrap/recalibration sample count (matches the 3-sample recal flow)
MIN_DISTINCT_GESTURES_READY = 2    # distinct gestures needed at MIN_SAMPLES before warm-up ends
PROTO_CAP = 8                       # max stored reference examples per gesture
ENROLL_SETTLE_WINDOWS = 3           # windows into a label run before enrolling (skips boundary contamination)
FAMILIARITY_UNFAMILIAR_RATIO = 3.0  # nearest-distance / typical-distance above this => "unfamiliar"
WARMUP_CONFIDENCE = 0.55
DEFAULT_GESTURE = "neutral"

WARMUP_NOTE = (
    "mock classifier (MOCK=1): warm-up — building per-subject reference "
    "profile from clean bootstrap windows, not yet scoring"
)


def _feature_vector(envelope: WindowEnvelope) -> list[float]:
    """Flatten RMS + normalized zero-crossing channels into one 24-dim vector."""
    rms = envelope["features"]["rms"]
    zc = envelope["features"]["zero_crossings"]
    return [float(v) for v in rms] + [float(v) / ZC_NORM_SCALE for v in zc]


def _distance(a: list[float], b: list[float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


class _GestureProfile:
    """One gesture's small, auditable reference-example set for one subject."""

    __slots__ = ("examples", "calibrated_at_ms", "note")

    def __init__(self) -> None:
        self.examples: list[list[float]] = []  # normalised feature vectors
        self.calibrated_at_ms: int = 0
        self.note: str = "bootstrap enrollment"

    @property
    def sample_count(self) -> int:
        return len(self.examples)

    @property
    def centroid(self) -> Optional[list[float]]:
        if not self.examples:
            return None
        n = len(self.examples)
        dim = len(self.examples[0])
        return [sum(v[i] for v in self.examples) / n for i in range(dim)]

    @property
    def dispersion(self) -> float:
        centroid = self.centroid
        if centroid is None:
            return 0.0
        return sum(_distance(v, centroid) for v in self.examples) / len(self.examples)

    def add(self, vec: list[float]) -> None:
        if len(self.examples) < PROTO_CAP:
            self.examples.append(vec)
            self.calibrated_at_ms = int(time.time() * 1000)

    def replace(self, vectors: list[list[float]], note: str) -> None:
        self.examples = list(vectors[:PROTO_CAP])
        self.calibrated_at_ms = int(time.time() * 1000)
        self.note = note

    def nearest_distance(self, vec: list[float]) -> float:
        return min(_distance(vec, e) for e in self.examples)

    def audit(self) -> dict:
        """Auditable snapshot: centroid, sample_count, dispersion, calibration metadata."""
        return {
            "centroid": self.centroid,
            "sample_count": self.sample_count,
            "dispersion": round(self.dispersion, 4),
            "calibrated_at_ms": self.calibrated_at_ms,
            "note": self.note,
        }


class _SubjectState:
    """Per-subject bootstrap + nearest-prototype state (process-lifetime, in memory)."""

    def __init__(self) -> None:
        self.profiles: dict[str, _GestureProfile] = {}
        self.norm_mean: Optional[list[float]] = None
        self.norm_std: Optional[list[float]] = None
        self.ready: bool = False
        self.typical_distance: float = 1.0

        self._bootstrap_pool: list[tuple[str, list[float]]] = []
        self._last_true_gesture: Optional[str] = None
        self._run_length: int = 0
        self._enrolled_this_run: bool = False

    # -- normalisation -----------------------------------------------------

    def _normalize(self, vec: list[float]) -> list[float]:
        assert self.norm_mean is not None and self.norm_std is not None
        return [(v - m) / s for v, m, s in zip(vec, self.norm_mean, self.norm_std)]

    def _freeze_normalization(self) -> None:
        pooled = [v for _, v in self._bootstrap_pool]
        n = len(pooled)
        dim = len(pooled[0])
        mean = [sum(v[i] for v in pooled) / n for i in range(dim)]
        var = [sum((v[i] - mean[i]) ** 2 for v in pooled) / n for i in range(dim)]
        std = [max(math.sqrt(x), 1e-6) for x in var]
        self.norm_mean = mean
        self.norm_std = std

        for gesture, raw_vec in self._bootstrap_pool:
            profile = self.profiles.setdefault(gesture, _GestureProfile())
            profile.add(self._normalize(raw_vec))

        examples = [e for p in self.profiles.values() for e in p.examples]
        nn_distances = []
        for i, vi in enumerate(examples):
            candidates = [_distance(vi, vj) for j, vj in enumerate(examples) if j != i]
            if candidates:
                nn_distances.append(min(candidates))
        self.typical_distance = (sum(nn_distances) / len(nn_distances)) if nn_distances else 1.0

        self.ready = True
        self._bootstrap_pool = []

    # -- enrollment ----------------------------------------------------------

    def _is_settled_sample(self, true_gesture: str) -> bool:
        """True on the ENROLL_SETTLE_WINDOWS-th consecutive window of a label
        run — skips the boundary-contaminated first windows of a run and
        enrolls at most once per run."""
        if true_gesture != self._last_true_gesture:
            self._run_length = 0
            self._enrolled_this_run = False
        else:
            self._run_length += 1
        self._last_true_gesture = true_gesture
        if self._run_length == ENROLL_SETTLE_WINDOWS and not self._enrolled_this_run:
            self._enrolled_this_run = True
            return True
        return False

    def observe_clean(self, true_gesture: str, raw_vec: list[float]) -> None:
        """Bootstrap/ongoing enrollment from a CLEAN, true-labeled window."""
        settled = self._is_settled_sample(true_gesture)

        if not self.ready:
            if settled:
                self._bootstrap_pool.append((true_gesture, raw_vec))
            counts: dict[str, int] = {}
            for gesture, _ in self._bootstrap_pool:
                counts[gesture] = counts.get(gesture, 0) + 1
            if sum(1 for c in counts.values() if c >= MIN_SAMPLES_PER_GESTURE) >= MIN_DISTINCT_GESTURES_READY:
                self._freeze_normalization()
            return

        if not settled:
            return
        profile = self.profiles.setdefault(true_gesture, _GestureProfile())
        profile.add(self._normalize(raw_vec))

    def enroll_confirmed(self, gesture: str, raw_vectors: list[list[float]]) -> None:
        """Recalibration hook: REPLACE *gesture*'s reference examples with
        exactly *raw_vectors* — a deterministic reset, not a blend."""
        if not self.ready:
            # No bootstrap has completed yet; seed it with these confirmed samples too.
            for v in raw_vectors:
                self._bootstrap_pool.append((gesture, v))
            counts: dict[str, int] = {}
            for g, _ in self._bootstrap_pool:
                counts[g] = counts.get(g, 0) + 1
            if sum(1 for c in counts.values() if c >= MIN_SAMPLES_PER_GESTURE) >= MIN_DISTINCT_GESTURES_READY:
                self._freeze_normalization()
            return

        normalized = [self._normalize(v) for v in raw_vectors]
        profile = self.profiles.setdefault(gesture, _GestureProfile())
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        profile.replace(normalized, note=f"recalibrated at {ts} from 3 confirmed windows")

    # -- inference -----------------------------------------------------------

    def predict(self, raw_vec: list[float]) -> ClassifierResponse:
        if not self.ready or not self.profiles:
            return {
                "gesture": DEFAULT_GESTURE,
                "confidence": WARMUP_CONFIDENCE,
                "drift_suspected": False,
                "notes": WARMUP_NOTE,
            }

        vec = self._normalize(raw_vec)
        ranked = sorted(
            (profile.nearest_distance(vec), gesture)
            for gesture, profile in self.profiles.items()
            if profile.examples
        )
        d_near, g_near = ranked[0]
        d_second = ranked[1][0] if len(ranked) > 1 else d_near + self.typical_distance

        margin = max(0.0, d_second - d_near)
        confidence = min(0.98, max(0.05, (margin / (d_second + d_near + 1e-6)) * 2))

        familiarity = d_near / max(self.typical_distance, 1e-6)
        drift_suspected = familiarity > FAMILIARITY_UNFAMILIAR_RATIO or confidence < 0.15
        if drift_suspected:
            confidence = min(confidence, 0.3)

        return {
            "gesture": g_near,
            "confidence": round(confidence, 3),
            "drift_suspected": drift_suspected,
            "notes": f"mock classifier (MOCK=1): nearest-prototype match against '{g_near}' reference profile",
        }


# ---------------------------------------------------------------------------
# Module-level per-subject registry
# ---------------------------------------------------------------------------
_subjects: dict[str, _SubjectState] = {}


def _get_state(subject: str) -> _SubjectState:
    return _subjects.setdefault(subject, _SubjectState())


def reset_subject(subject: str) -> None:
    """Drop all learned state for *subject*, forcing a fresh bootstrap.

    Not part of the WebSocket/HTTP contract — a test/demo utility only.
    """
    _subjects.pop(subject, None)


def get_profile_snapshot(subject: str) -> dict:
    """Auditable snapshot of *subject*'s per-gesture reference profile:
    centroid, sample_count, dispersion, and calibration metadata per
    gesture. Not part of the WebSocket/HTTP contract — for inspection,
    debugging, and tests only.
    """
    state = _subjects.get(subject)
    if state is None:
        return {"ready": False, "gestures": {}}
    return {
        "ready": state.ready,
        "gestures": {gesture: profile.audit() for gesture, profile in state.profiles.items()},
    }


def enroll_confirmed_samples(subject: str, gesture: str, envelopes: list[WindowEnvelope]) -> None:
    """agent.recalibrator's hook: replace *gesture*'s reference examples with
    the confirmed *envelopes* it just collected (fresh, undrifted windows of
    the confused gesture) — see the module docstring's Recalibration section.
    A no-op if *envelopes* is empty.
    """
    vectors = [_feature_vector(e) for e in envelopes]
    if vectors:
        _get_state(subject).enroll_confirmed(gesture, vectors)


async def classify(
    envelope: WindowEnvelope,
    *,
    history: list[ClassifierResponse] | None = None,
    profile: dict | None = None,
    cost_tracker: CostTracker | None = None,
    api_key: str | None = None,
) -> ClassifierResponse:
    """
    Classify *envelope* from its RMS/zero-crossing features alone — no
    network call, no spectrogram image, never envelope["true_gesture"] as
    the prediction. See the module docstring for the bootstrap / inference /
    recalibration design.

    `history`, `profile`, `cost_tracker`, and `api_key` are accepted for
    interface compatibility with agent.classifier.classify but are unused:
    this module keeps its own per-subject reference profile in memory
    (see get_profile_snapshot) rather than reading gesture_profiles/*.json.

    Parameters
    ----------
    envelope:
        A fully populated WindowEnvelope dict.

    Returns
    -------
    ClassifierResponse with gesture, confidence, drift_suspected, and notes.
    """
    state = _get_state(envelope["subject"])
    raw_vec = _feature_vector(envelope)

    result = state.predict(raw_vec)

    true_gesture = envelope.get("true_gesture")
    if true_gesture in POSE_SET:
        state.observe_clean(true_gesture, raw_vec)

    return result
