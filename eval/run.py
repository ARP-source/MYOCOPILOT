"""
eval.run — offline end-to-end evaluation harness.

Runs 3 subjects x {clean, drift, drift_plus_recalibration} entirely offline
and writes structured results to eval/results.json. Never touches the
network and never requires OPENROUTER_API_KEY: classification always goes
through agent.mock_classifier directly (the same deterministic per-subject
nearest-prototype classifier MOCK=1 uses in the live app), regardless of
what the MOCK environment variable is set to in the calling shell, so this
script is safe to run anywhere.

All results here come from myosignal.loaders.make_synthetic_session's
structured, deliberately-engineered synthetic EMG-like signals
("synthetic_demo_data") — never real NinaPro/emg2pose recordings unless
files are dropped under data/. Nothing here is clinical validation.

Data
----
Looks for real NinaPro DB2 (.mat) or emg2pose (.h5/.hdf5) files under data/
for up to 3 subjects. Any subject slot without a real file gets a
deterministic synthetic session instead (myosignal.loaders.make_synthetic_session),
and its "data_source" field is set to "synthetic_demo_data" so downstream
consumers (eval/report.py, README) can never mistake it for real data.

Conditions
----------
clean
    No drift. Whole session classified as-is. baseline_accuracy is the
    overall accuracy; post_drift/post_recalibration are null.

drift
    For DRIFT_MODE == "electrode_shift" (the default), a structured
    synthetic electrode-shift STRESS TEST — myosignal.player's
    apply_electrode_shift_stress: a deterministic whole-12-channel
    permutation, a fixed per-channel gain (0.45x-1.70x), small
    channel-dependent baseline wander/contact noise, and two channels
    losing most contact — injected starting ~35% into the session. (Other
    DRIFT_MODE values still go through the milder, localized
    myosignal.drift.inject_drift.) Accuracy is reported before (baseline)
    and after (post_drift) the injection point. No recalibration happens,
    so post_recalibration is null.

drift_plus_recalibration
    Same drift injection, but once agent.drift_detector.check_drift() flags
    the dominant confusion pair, a recalibration is simulated: 3 example
    windows of the confused gesture are pulled from the ORIGINAL undrifted
    session (mirroring agent.recalibrator.recalibrate(), which always starts
    a *fresh* myosignal stream to collect both the reference examples and
    the post-recalibration windows — never the still-drifted live stream)
    and fed to agent.mock_classifier.enroll_confirmed_samples(), which
    replaces the confused gesture's reference profile exactly as the real
    conversational recalibration flow does. Classification then continues
    from clean windows for the rest of the session.
    gesture_accuracy_before_after_recalibration reports the confused
    gesture's OWN before/after accuracy (the calibrated-target recovery);
    post_recalibration_accuracy is the OVERALL rolling accuracy across all
    gestures in that same tail — these can differ (see eval/report.py).

Run with:
    python -m eval.run
    MOCK=1 python -m eval.run    (MOCK is accepted but irrelevant — see above)
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # avoid mojibake on Windows consoles (cp1252 default)
except (AttributeError, ValueError):
    pass

os.environ.setdefault("MOCK", "1")  # belt-and-suspenders; classify() below never calls out anyway

from sklearn.metrics import precision_recall_fscore_support

from myosignal.types import POSE_SET
from myosignal.windows import WindowBatch, STRIDE_MS, compute_rms, compute_zero_crossings
from myosignal.drift import inject_drift
from myosignal.player import apply_electrode_shift_stress

from agent import mock_classifier
from agent.mock_classifier import classify as mock_classify
from agent.drift_detector import DriftDetector, check_drift
from agent.cost_tracker import COST_PER_INPUT_TOKEN_USD, COST_PER_OUTPUT_TOKEN_USD

from eval._common import (
    N_SUBJECTS,
    discover_subjects,
    data_source_label,
    load_and_segment,
)

CONDITIONS = ["clean", "drift", "drift_plus_recalibration"]
DRIFT_MODE = "electrode_shift"
DRIFT_FRACTION = 0.35          # drift starts 35% of the way through the session
N_RECAL_EXAMPLES = 3           # matches web/server.py's live recalibration flow
ROLLING_WINDOW = 20            # matches agent.drift_detector.DriftDetector default

# Rough per-call token footprint of a real GLM-5.3-Flash classify() request
# (one 512px-wide multimodal spectrogram image + short text context/response).
# This run never calls the API — these are estimates for planning API cost,
# not measurements. See agent/cost_tracker.py for the pricing constants.
EST_INPUT_TOKENS_PER_WINDOW = 1150
EST_OUTPUT_TOKENS_PER_WINDOW = 40
EST_INTERVIEW_QUESTION_TOKENS = 220  # one extra text-only call per recalibration event

RESULTS_PATH = Path("eval/results.json")


def _t_ms(base_ms: int, window_index: int) -> int:
    return base_ms + window_index * STRIDE_MS


def _gesture_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    if not y_true:
        return {g: {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0} for g in POSE_SET}
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=POSE_SET, zero_division=0
    )
    return {
        g: {
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "support": int(support[i]),
        }
        for i, g in enumerate(POSE_SET)
    }


def _confusion_matrix(y_true: list[str], y_pred: list[str]) -> list[list[int]]:
    idx = {g: i for i, g in enumerate(POSE_SET)}
    matrix = [[0] * len(POSE_SET) for _ in POSE_SET]
    for t, p in zip(y_true, y_pred):
        matrix[idx[t]][idx[p]] += 1
    return matrix


def _accuracy(y_true: list[str], y_pred: list[str]) -> float | None:
    if not y_true:
        return None
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return round(correct / len(y_true), 4)


async def _classify_batches(
    batches: list[WindowBatch],
    subject: str,
    base_t_ms: int,
    drift_start_index: int | None,
    fs: int,
    recalibrate: bool,
) -> dict:
    """
    Classify *batches* window-by-window, optionally injecting drift at
    *drift_start_index* and (if *recalibrate*) simulating recalibration on
    the first detected drift. Returns a dict of raw per-window traces plus
    event timestamps; the caller turns this into the results.json record.
    """
    detector = DriftDetector(window_size=ROLLING_WINDOW)
    # A second detector, only instantiated once drift injection actually
    # starts, so the accuracy-threshold trigger judges drift-affected
    # windows on their own terms — not on whatever rolling accuracy the
    # classifier accumulated during its own early bootstrap/warm-up phase
    # (agent.mock_classifier only starts real nearest-prototype scoring
    # after a short warm-up; that warm-up's forced "neutral" guesses would
    # otherwise look like drift to a detector monitoring from window 0).
    check_detector: DriftDetector | None = None
    y_true: list[str] = []
    y_pred: list[str] = []
    rolling_series: list[dict] = []
    n_classified = 0
    n_recal_calls = 0

    drift_event: dict | None = None
    recal_event: dict | None = None
    recal_window_index: int | None = None

    i = 0
    n = len(batches)
    while i < n:
        batch = batches[i]
        if drift_start_index is not None and i == drift_start_index:
            check_detector = DriftDetector(window_size=ROLLING_WINDOW)
        use_clean = (
            drift_start_index is None
            or i < drift_start_index
            or (recal_window_index is not None and i >= recal_window_index)
        )

        if use_clean:
            window = batch.window
            rms = batch.rms
            zc = batch.zero_crossings
        else:
            if DRIFT_MODE == "electrode_shift":
                # Severe, deterministic whole-array electrode-cap remap — see
                # myosignal.player.apply_electrode_shift_stress's module comment.
                # A structured synthetic electrode-shift STRESS TEST, not a
                # clinical/biomechanical model.
                window = apply_electrode_shift_stress(batch.window, fs)
            else:
                elapsed_s = (i - drift_start_index) * STRIDE_MS / 1000.0
                window = inject_drift(batch.window, mode=DRIFT_MODE, elapsed_s=elapsed_s, fs=fs)
            rms = compute_rms(window)
            zc = compute_zero_crossings(window)

        true_gesture = batch.label_name if batch.label_name in POSE_SET else None
        envelope = {
            "t_ms": _t_ms(base_t_ms, i),
            "subject": subject,
            "true_gesture": true_gesture,
            "spectrogram_png_b64": "",  # mock_classifier never looks at this
            "features": {"rms": rms.tolist(), "zero_crossings": zc.tolist()},
        }

        pred = await mock_classify(envelope)
        n_classified += 1

        if true_gesture is not None:
            y_true.append(true_gesture)
            y_pred.append(pred["gesture"])

        detector.update(true_gesture, pred)
        rolling_series.append(
            {"window_index": i, "t_ms": envelope["t_ms"], "acc": detector.rolling_accuracy()}
        )
        if check_detector is not None:
            check_detector.update(true_gesture, pred)

        if check_detector is not None and drift_event is None:
            pair = check_drift(check_detector)
            if pair is not None:
                target_gesture, confused_with = pair
                drift_event = {
                    "window_index": i,
                    "t_ms": envelope["t_ms"],
                    "target_gesture": target_gesture,
                    "confused_with": confused_with,
                }
                if recalibrate:
                    # Pull fresh (undrifted) examples of target_gesture, exactly as
                    # agent.recalibrator.collect_recalibration_samples() does against
                    # a brand-new myosignal stream.
                    examples = [
                        b for b in batches[:drift_start_index] if b.label_name == target_gesture
                    ][:N_RECAL_EXAMPLES]
                    n_recal_calls = len(examples)
                    recal_window_index = i + 1
                    recal_event = {
                        "window_index": recal_window_index,
                        "t_ms": _t_ms(base_t_ms, recal_window_index),
                        "gesture": target_gesture,
                        "n_examples_collected": n_recal_calls,
                    }
                    if examples:
                        # Mirror agent.recalibrator.recalibrate()'s
                        # enroll_confirmed_samples() call, so this simulated
                        # recalibration actually replaces target_gesture's
                        # reference profile the same way the real
                        # conversational flow does.
                        confirmed_envelopes = [
                            {
                                "t_ms": _t_ms(base_t_ms, idx),
                                "subject": subject,
                                "true_gesture": target_gesture,
                                "spectrogram_png_b64": "",
                                "features": {
                                    "rms": b.rms.tolist(),
                                    "zero_crossings": b.zero_crossings.tolist(),
                                },
                            }
                            for idx, b in enumerate(examples)
                        ]
                        mock_classifier.enroll_confirmed_samples(
                            subject, target_gesture, confirmed_envelopes
                        )

        i += 1

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "rolling_series": rolling_series,
        "n_classified": n_classified,
        "n_recal_calls": n_recal_calls,
        "drift_event": drift_event,
        "recal_event": recal_event,
    }


def _segment_accuracy(y_true, y_pred, indices_true, lo, hi) -> float | None:
    """Accuracy over the sub-range of labeled windows whose original batch index falls in [lo, hi)."""
    pairs = [
        (t, p) for t, p, idx in zip(y_true, y_pred, indices_true) if lo <= idx < hi
    ]
    if not pairs:
        return None
    correct = sum(1 for t, p in pairs if t == p)
    return round(correct / len(pairs), 4)


async def _run_subject_condition(
    subject_id: str, batches: list[WindowBatch], fs: int, condition: str, seed: int
) -> dict:
    base_t_ms = int(time.time() * 1000)
    total_windows = len(batches)
    drift_start_index = (
        None if condition == "clean" else int(total_windows * DRIFT_FRACTION)
    )

    trace = await _classify_batches(
        batches,
        subject=subject_id,
        base_t_ms=base_t_ms,
        drift_start_index=drift_start_index,
        fs=fs,
        recalibrate=(condition == "drift_plus_recalibration"),
    )

    # Track each labeled window's original batch index so we can split
    # accuracy into baseline / post-drift / post-recalibration segments.
    labeled_indices = [
        i for i, b in enumerate(batches) if b.label_name in POSE_SET
    ]

    overall_accuracy = _accuracy(trace["y_true"], trace["y_pred"])

    if condition == "clean":
        baseline_accuracy = overall_accuracy
        post_drift_accuracy = None
        post_recalibration_accuracy = None
    else:
        baseline_accuracy = _segment_accuracy(
            trace["y_true"], trace["y_pred"], labeled_indices, 0, drift_start_index
        )
        recal_idx = (
            trace["recal_event"]["window_index"] if trace["recal_event"] else total_windows
        )
        post_drift_accuracy = _segment_accuracy(
            trace["y_true"], trace["y_pred"], labeled_indices, drift_start_index, recal_idx
        )
        if condition == "drift_plus_recalibration":
            post_recalibration_accuracy = _segment_accuracy(
                trace["y_true"], trace["y_pred"], labeled_indices, recal_idx, total_windows
            )
        else:
            post_recalibration_accuracy = None

    # Per-gesture pre/post-recalibration accuracy, for the "before vs after" chart.
    gesture_pre_post: dict | None = None
    if condition == "drift_plus_recalibration" and trace["recal_event"] is not None:
        recal_idx = trace["recal_event"]["window_index"]
        gesture_pre_post = {}
        for g in POSE_SET:
            pre_pairs = [
                (t, p)
                for t, p, idx in zip(trace["y_true"], trace["y_pred"], labeled_indices)
                if drift_start_index <= idx < recal_idx and t == g
            ]
            post_pairs = [
                (t, p)
                for t, p, idx in zip(trace["y_true"], trace["y_pred"], labeled_indices)
                if idx >= recal_idx and t == g
            ]
            gesture_pre_post[g] = {
                "before": round(sum(1 for t, p in pre_pairs if t == p) / len(pre_pairs), 4)
                if pre_pairs
                else None,
                "after": round(sum(1 for t, p in post_pairs if t == p) / len(post_pairs), 4)
                if post_pairs
                else None,
            }

    n_windows = trace["n_classified"]
    n_recal_interview_calls = 1 if trace["recal_event"] else 0
    est_tokens = n_windows * (EST_INPUT_TOKENS_PER_WINDOW + EST_OUTPUT_TOKENS_PER_WINDOW)
    est_tokens += n_recal_interview_calls * EST_INTERVIEW_QUESTION_TOKENS
    est_cost = (
        n_windows * EST_INPUT_TOKENS_PER_WINDOW * COST_PER_INPUT_TOKEN_USD
        + n_windows * EST_OUTPUT_TOKENS_PER_WINDOW * COST_PER_OUTPUT_TOKEN_USD
        + n_recal_interview_calls * EST_INTERVIEW_QUESTION_TOKENS * COST_PER_INPUT_TOKEN_USD
    )

    return {
        "subject": subject_id,
        "condition": condition,
        "n_windows": n_windows,
        "baseline_accuracy": baseline_accuracy,
        "post_drift_accuracy": post_drift_accuracy,
        "post_recalibration_accuracy": post_recalibration_accuracy,
        "drift_mode": DRIFT_MODE if condition != "clean" else None,
        "drift_injected_at_window_index": drift_start_index,
        "drift_injected_at_t_ms": _t_ms(base_t_ms, drift_start_index) if drift_start_index is not None else None,
        "drift_detected_at_t_ms": trace["drift_event"]["t_ms"] if trace["drift_event"] else None,
        "drift_detected_at_window_index": (
            trace["drift_event"]["window_index"] if trace["drift_event"] else None
        ),
        "confused_pair": (
            [trace["drift_event"]["target_gesture"], trace["drift_event"]["confused_with"]]
            if trace["drift_event"]
            else None
        ),
        "recalibration_at_t_ms": trace["recal_event"]["t_ms"] if trace["recal_event"] else None,
        "recalibration_at_window_index": (
            trace["recal_event"]["window_index"] if trace["recal_event"] else None
        ),
        "recalibration_examples_collected": trace["n_recal_calls"],
        "gesture_accuracy_before_after_recalibration": gesture_pre_post,
        "gesture_metrics": _gesture_metrics(trace["y_true"], trace["y_pred"]),
        "confusion_matrix": _confusion_matrix(trace["y_true"], trace["y_pred"]),
        "rolling_accuracy_series": trace["rolling_series"],
        "estimated_tokens": est_tokens,
        "estimated_cost_usd": round(est_cost, 6),
    }


async def run() -> dict:
    subjects = discover_subjects(N_SUBJECTS)
    runs: list[dict] = []

    for seed, descriptor in enumerate(subjects, start=1):
        raw, batches = load_and_segment(descriptor, seed=seed)
        source_label = data_source_label(descriptor)
        for condition in CONDITIONS:
            record = await _run_subject_condition(
                descriptor["subject_id"], batches, raw.fs, condition, seed
            )
            record["data_source"] = source_label
            runs.append(record)
            print(
                f"  {descriptor['subject_id']:>22} [{source_label:>19}] "
                f"{condition:<26} n={record['n_windows']:<4} "
                f"baseline={record['baseline_accuracy']} "
                f"post_drift={record['post_drift_accuracy']} "
                f"post_recal={record['post_recalibration_accuracy']}"
            )

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "mock_mode": True,
        "note": (
            "Classification always uses agent.mock_classifier (network-free "
            "RMS heuristic), independent of the MOCK env var, so this file "
            "never reflects real GLM-5.3-Flash accuracy. estimated_tokens/"
            "estimated_cost_usd are projections for the real API, not "
            "measurements — see eval/run.py module docstring."
        ),
        "pose_set": POSE_SET,
        "conditions": CONDITIONS,
        "drift_mode": DRIFT_MODE,
        "runs": runs,
    }
    return results


def main() -> None:
    print(f"MyoCopilot offline evaluation — {N_SUBJECTS} subjects x {len(CONDITIONS)} conditions")
    results = asyncio.run(run())
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {RESULTS_PATH} ({len(results['runs'])} subject/condition records)")


if __name__ == "__main__":
    main()
