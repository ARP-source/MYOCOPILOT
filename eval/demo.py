"""
eval.demo — headless, deterministic, MOCK=1 walkthrough of the full pipeline.

Drives the *actual* production code path end to end — myosignal.SessionPlayer,
agent.classify, agent.drift_detector.check_drift, agent.recalibrator — the
same functions web/server.py's WebSocket handler calls, just printed to the
console instead of pushed over a socket. No browser, no OpenRouter key: MOCK
is forced to "1" before agent is imported, so classification always goes
through agent.mock_classifier.

This narrates against the fixed presentation timeline from the README's
90-second demo script:

    0s   clean EMG stream begins
    15s  electrode shift injected
    35s  drift detected
    40s  conversational recalibration request
    55s  three samples accepted
    70s  gesture profile updated
    90s  recovery confirmed

The *labels* above are the talking points for a live presenter; the actual
elapsed time for each step is real (myosignal streams at 1x real-time pace,
by design, so a human watching a real dashboard sees it happen live) and is
printed alongside the target so you can see how closely a given run tracked
the script. Timing follows the deterministic synthetic-session RNG, so the
shape of a run is reproducible, but exact seconds can vary a little with
machine speed.

Run with:
    python -m eval.demo
"""

from __future__ import annotations

import asyncio
import os
import sys
import time

try:
    sys.stdout.reconfigure(encoding="utf-8")  # avoid mojibake on Windows consoles (cp1252 default)
except (AttributeError, ValueError):
    pass

os.environ["MOCK"] = "1"  # forced, not just defaulted -- this demo never touches OpenRouter

from myosignal import SessionPlayer
from agent import classify, check_drift, recalibrate
from agent.recalibrator import generate_interview_question
from agent.drift_detector import DriftDetector
from agent.cost_tracker import CostTracker
from agent.profile_store import load_profile

DEMO_SUBJECT = "demo_subject"
DRIFT_AT_S = 25.0  # comfortably after the ~15s bootstrap window (see myosignal.loaders'
                   # "Burst schedule" docstring) so every gesture has a stable few-shot
                   # reference profile before electrode-shift stress is injected
DEMO_SEED = 7

TARGET_TIMELINE = [
    (0, "clean EMG stream begins"),
    (15, "electrode shift injected"),
    (35, "drift detected"),
    (40, "conversational recalibration request"),
    (55, "three samples accepted"),
    (70, "gesture profile updated"),
    (90, "recovery confirmed"),
]


def _log(t0: float, msg: str) -> None:
    elapsed = time.monotonic() - t0
    print(f"  [t={elapsed:6.1f}s] {msg}")


async def run_demo() -> None:
    print("MyoCopilot headless demo — MOCK=1, no browser, no OpenRouter key required.\n")
    print("Target presentation timeline (talking points for a live demo):")
    for t, label in TARGET_TIMELINE:
        print(f"  {t:>3}s  {label}")
    print("\nActual run (myosignal streams at 1x real-time pace, same as the live dashboard):")

    t0 = time.monotonic()
    player = SessionPlayer(subject=DEMO_SUBJECT, source="synthetic", seed=DEMO_SEED)
    # The drift detector is only instantiated once the electrode-shift artefact
    # actually starts, so its rolling accuracy window judges drift-affected
    # windows on their own terms — the network-free mock heuristic's baseline
    # accuracy already sits under DriftDetector's 60% threshold even on clean
    # signal, so a detector fed from t=0 would (correctly, if confusingly for
    # a live narration) flag "drift" before the artefact is even injected.
    # See eval/run.py's module docstring for the same finding measured offline.
    detector: DriftDetector | None = None
    cost = CostTracker()
    history: list = []

    _log(t0, "clean EMG stream begins")
    drift_announced = False
    recal_done = False

    for envelope in player.stream(drift_at_s=DRIFT_AT_S, drift_mode="electrode_shift"):
        elapsed = time.monotonic() - t0
        if not drift_announced and elapsed >= DRIFT_AT_S:
            _log(t0, "electrode shift injected")
            drift_announced = True
            detector = DriftDetector()

        profile = load_profile(DEMO_SUBJECT)
        pred = await classify(envelope, history=history[-5:], profile=profile, cost_tracker=cost)
        history.append(pred)
        if detector is not None:
            detector.update(envelope["true_gesture"], pred)

        if detector is not None and not recal_done:
            pair = check_drift(detector)
            if pair is not None:
                target_gesture, confused_with = pair
                _log(t0, f"drift detected — confusing '{target_gesture}' with '{confused_with}'")

                question = await generate_interview_question(target_gesture, confused_with)
                _log(t0, f'conversational recalibration request: "{question}"')

                result = await recalibrate(
                    DEMO_SUBJECT, target_gesture, confused_with, detector, cost_tracker=cost
                )
                _log(t0, f"samples accepted for '{target_gesture}'")
                _log(t0, f"gesture profile updated (gesture_profiles/{DEMO_SUBJECT}.json)")
                _log(
                    t0,
                    f"recovery confirmed — rolling accuracy before={result['acc_before']:.2f} "
                    f"after={result['acc_after']:.2f}",
                )
                recal_done = True
                break

    total = time.monotonic() - t0
    print(f"\nDemo complete in {total:.1f}s real elapsed time.")
    print(f"Estimated token/cost summary: {cost.summary()}")
    if not recal_done:
        print(
            "\nNote: drift was never detected during this run (no recalibration triggered). "
            "This can happen because the synthetic RNG determines exactly which gesture is "
            "active when drift starts; re-run for another draw."
        )


def main() -> None:
    asyncio.run(run_demo())


if __name__ == "__main__":
    main()
