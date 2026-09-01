"""
eval.baselines — honest classical baseline: k-NN on RMS features.

Trains a small k-nearest-neighbours classifier (scikit-learn) on each
subject's first two repetitions using the 12-channel RMS feature vector
already computed by myosignal.windows.segment(), and evaluates on the
remaining repetitions. For subjects with no repetition structure (e.g. a
synthetic session where large stretches sit at repetition 0 / rest), a
deterministic 60/40 chronological split is used instead so the script never
silently produces an empty split.

This is a sanity-check baseline, not a competitor: it exists so nobody can
claim MyoCopilot's numbers "beat" a classifier without a classical baseline
sitting next to them. Nothing here is validated against real clinical data,
and neither this baseline nor MyoCopilot's mock/agent path should be read as
clinically meaningful — see the data_source column.

Run with:
    python -m eval.baselines
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")  # avoid mojibake on Windows consoles (cp1252 default)
except (AttributeError, ValueError):
    pass

from sklearn.neighbors import KNeighborsClassifier

from myosignal.types import POSE_SET
from eval._common import N_SUBJECTS, discover_subjects, data_source_label, load_and_segment

RESULTS_PATH = Path("eval/results.json")
OUT_PATH = Path("eval/baseline_results.json")
K_NEIGHBORS = 5


def _split_train_eval(batches):
    """Repetitions 1-2 -> train, repetitions >=3 -> eval; falls back to a
    deterministic 60/40 chronological split if that leaves either side empty
    (e.g. a synthetic session with few/no repetition boundaries)."""
    labeled = [b for b in batches if b.label_name in POSE_SET]
    train = [b for b in labeled if b.rep in (1, 2)]
    eval_ = [b for b in labeled if b.rep >= 3]
    if train and eval_:
        return train, eval_, "repetitions 1-2 train / repetitions >=3 eval"
    split = int(len(labeled) * 0.6)
    return labeled[:split], labeled[split:], "chronological 60/40 split (fallback: no rep structure)"


def _mock_accuracy_for(subject_id: str, results: dict | None) -> float | None:
    if not results:
        return None
    for run in results.get("runs", []):
        if run["subject"] == subject_id and run["condition"] == "clean":
            return run["baseline_accuracy"]
    return None


def run() -> dict:
    subjects = discover_subjects(N_SUBJECTS)
    results = json.loads(RESULTS_PATH.read_text()) if RESULTS_PATH.exists() else None

    rows = []
    for seed, descriptor in enumerate(subjects, start=1):
        _, batches = load_and_segment(descriptor, seed=seed)
        train, eval_, split_desc = _split_train_eval(batches)

        if train and eval_:
            X_train = [b.rms.tolist() for b in train]
            y_train = [b.label_name for b in train]
            X_eval = [b.rms.tolist() for b in eval_]
            y_eval = [b.label_name for b in eval_]

            knn = KNeighborsClassifier(n_neighbors=min(K_NEIGHBORS, len(train)))
            knn.fit(X_train, y_train)
            preds = knn.predict(X_eval)
            correct = sum(1 for t, p in zip(y_eval, preds) if t == p)
            knn_accuracy = round(correct / len(y_eval), 4)
        else:
            knn_accuracy = None

        rows.append(
            {
                "subject": descriptor["subject_id"],
                "data_source": data_source_label(descriptor),
                "split": split_desc,
                "n_train_windows": len(train),
                "n_eval_windows": len(eval_),
                "knn_rms_accuracy": knn_accuracy,
                "myocopilot_mock_clean_accuracy": _mock_accuracy_for(descriptor["subject_id"], results),
            }
        )

    return {
        "note": (
            "Sanity-check baseline only — not a claim of clinical accuracy or of "
            "superiority over the classical k-NN method. Compare only within the "
            "same data_source; synthetic_demo_data rows are not real EMG."
        ),
        "k_neighbors": K_NEIGHBORS,
        "rows": rows,
    }


def _print_table(report: dict) -> None:
    headers = ["subject", "data_source", "n_train", "n_eval", "knn_acc", "myocopilot_acc"]
    widths = [22, 19, 8, 8, 9, 15]
    print("  ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  ".join("-" * w for w in widths))
    for r in report["rows"]:
        cells = [
            r["subject"],
            r["data_source"],
            str(r["n_train_windows"]),
            str(r["n_eval_windows"]),
            "n/a" if r["knn_rms_accuracy"] is None else f"{r['knn_rms_accuracy']:.4f}",
            "n/a" if r["myocopilot_mock_clean_accuracy"] is None else f"{r['myocopilot_mock_clean_accuracy']:.4f}",
        ]
        print("  ".join(c.ljust(w) for c, w in zip(cells, widths)))
    print(f"\n{report['note']}")


def main() -> None:
    report = run()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _print_table(report)
    print(f"\nWrote {OUT_PATH}")


if __name__ == "__main__":
    main()
