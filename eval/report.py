"""
eval.report — generate publication-quality figures from eval/results.json.

Reads eval/results.json (produced by `python -m eval.run`) and, if present,
eval/baseline_results.json (produced by `python -m eval.baselines`), and
writes dark-on-light PNG figures suitable for slides to eval/report/:

    confusion_matrices.png       per-subject x per-condition confusion matrices
    accuracy_vs_time.png         rolling accuracy with drift/recalibration markers
    recalibration_before_after.png   per-gesture accuracy before vs after recalibration
    cost_token_summary.png       token/cost table across all subject/condition runs
    baseline_comparison.png      classical k-NN baseline vs MyoCopilot mock path

Every figure's title states plainly whether the underlying data is real or
synthetic_demo_data — never silently omitted. Drift figures additionally
describe the "electrode_shift" condition as a structured synthetic
electrode-shift STRESS TEST (myosignal.player.apply_electrode_shift_stress):
not a clinical or biomechanical model, and not clinical validation of
anything.

Run with:
    python -m eval.report
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_PATH = Path("eval/results.json")
BASELINE_PATH = Path("eval/baseline_results.json")
OUT_DIR = Path("eval/report")
DPI = 160

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#1a1a1a",
        "text.color": "#1a1a1a",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
    }
)


def _load_results() -> dict:
    if not RESULTS_PATH.exists():
        raise SystemExit(f"{RESULTS_PATH} not found — run `python -m eval.run` first.")
    return json.loads(RESULTS_PATH.read_text())


def _load_baselines() -> dict | None:
    if not BASELINE_PATH.exists():
        return None
    return json.loads(BASELINE_PATH.read_text())


def _subjects(results: dict) -> list[str]:
    seen = []
    for run in results["runs"]:
        if run["subject"] not in seen:
            seen.append(run["subject"])
    return seen


def _find(results: dict, subject: str, condition: str) -> dict | None:
    for run in results["runs"]:
        if run["subject"] == subject and run["condition"] == condition:
            return run
    return None


def _data_source_suffix(results: dict, subject: str) -> str:
    run = _find(results, subject, "clean")
    return run["data_source"] if run else "unknown"


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


def plot_confusion_matrices(results: dict, out_path: Path) -> None:
    subjects = _subjects(results)
    conditions = results["conditions"]
    labels = results["pose_set"]

    fig, axes = plt.subplots(
        len(subjects), len(conditions), figsize=(4.2 * len(conditions), 4.2 * len(subjects))
    )
    axes = np.atleast_2d(axes)

    for r, subject in enumerate(subjects):
        for c, condition in enumerate(conditions):
            ax = axes[r][c]
            run = _find(results, subject, condition)
            matrix = np.array(run["confusion_matrix"]) if run else np.zeros((len(labels), len(labels)))
            im = ax.imshow(matrix, cmap="Blues")
            ax.set_xticks(range(len(labels)))
            ax.set_yticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=90, fontsize=6)
            ax.set_yticklabels(labels, fontsize=6)
            src = run["data_source"] if run else "unknown"
            title = f"{subject}\n{condition} [{src}]"
            ax.set_title(title, fontsize=8)
            if r == len(subjects) - 1:
                ax.set_xlabel("predicted")
            if c == 0:
                ax.set_ylabel("true")
            vmax = max(matrix.max(), 1)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    v = int(matrix[i, j])
                    if v:
                        color = "white" if v > vmax * 0.6 else "#222222"
                        ax.text(j, i, str(v), ha="center", va="center", fontsize=5, color=color)

    fig.suptitle(
        "MyoCopilot offline evaluation — confusion matrices (agent.mock_classifier, MOCK path)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_vs_time(results: dict, out_path: Path) -> None:
    subjects = _subjects(results)
    fig, axes = plt.subplots(len(subjects), 1, figsize=(9, 3.2 * len(subjects)), squeeze=False)

    for r, subject in enumerate(subjects):
        ax = axes[r][0]
        run = _find(results, subject, "drift_plus_recalibration")
        if run is None or not run["rolling_accuracy_series"]:
            ax.set_visible(False)
            continue
        series = run["rolling_accuracy_series"]
        xs = [p["window_index"] for p in series]
        ys = [p["acc"] for p in series]
        ax.plot(xs, ys, color="#1f6feb", linewidth=1.4, label="rolling accuracy (window=20)")

        if run["drift_injected_at_window_index"] is not None:
            ax.axvline(run["drift_injected_at_window_index"], color="#d9822b", linestyle="--", linewidth=1.2,
                        label="drift injected")
        if run["drift_detected_at_window_index"] is not None:
            ax.axvline(run["drift_detected_at_window_index"], color="#cf222e", linestyle=":", linewidth=1.2,
                        label="drift detected")
        if run["recalibration_at_window_index"] is not None:
            ax.axvline(run["recalibration_at_window_index"], color="#1a7f37", linestyle="-.", linewidth=1.2,
                        label="recalibration")

        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("window index (50 ms stride)")
        ax.set_ylabel("rolling accuracy")
        ax.set_title(f"{subject} [{run['data_source']}] — drift_plus_recalibration", fontsize=10)
        ax.legend(fontsize=7, loc="lower left")

    fig.suptitle(
        "Accuracy vs. time — structured synthetic electrode-shift stress test\n"
        "(drift and recalibration markers; synthetic_demo_data, not clinical validation)",
        fontsize=12.5,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_recalibration_before_after(results: dict, out_path: Path) -> None:
    """
    Per-gesture accuracy before vs. after recalibration — explicitly
    distinguishing two different recovery metrics that are easy to conflate:

      OVERALL rolling accuracy  — post_drift_accuracy -> post_recalibration_accuracy,
                                  averaged across every gesture in that window.
      CALIBRATED TARGET gesture — the specific confused gesture's own before/after
                                  accuracy (gesture_accuracy_before_after_recalibration),
                                  i.e. what the 3-sample recalibration flow actually
                                  targeted and replaced the reference profile for.

    The target gesture typically recovers further/faster than the overall
    average, since recalibration only replaces *its* profile — see the
    boxed summary and the highlighted bar pair on each subject's chart.
    """
    subjects = _subjects(results)
    labels = results["pose_set"]

    fig, axes = plt.subplots(len(subjects), 1, figsize=(9, 3.9 * len(subjects)), squeeze=False)

    for r, subject in enumerate(subjects):
        ax = axes[r][0]
        run = _find(results, subject, "drift_plus_recalibration")
        pre_post = run.get("gesture_accuracy_before_after_recalibration") if run else None
        if not pre_post:
            ax.text(0.5, 0.5, "no recalibration event recorded", ha="center", va="center")
            ax.set_axis_off()
            continue

        target_gesture = run["confused_pair"][0] if run.get("confused_pair") else None

        before = [pre_post[g]["before"] if pre_post[g]["before"] is not None else 0.0 for g in labels]
        after = [pre_post[g]["after"] if pre_post[g]["after"] is not None else 0.0 for g in labels]
        x = np.arange(len(labels))
        width = 0.38
        bars_before = ax.bar(x - width / 2, before, width, label="before recalibration", color="#cf222e")
        bars_after = ax.bar(x + width / 2, after, width, label="after recalibration", color="#1a7f37")

        if target_gesture in labels:
            ti = labels.index(target_gesture)
            for bar in (bars_before[ti], bars_after[ti]):
                bar.set_edgecolor("#1f6feb")
                bar.set_linewidth(2.5)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_ylim(0, 1.18)
        ax.set_ylabel("accuracy")
        ax.set_title(f"{subject} [{run['data_source']}] — blue outline = calibrated target gesture", fontsize=9.5)
        ax.legend(fontsize=8, loc="upper left")

        target_before = pre_post.get(target_gesture, {}).get("before") if target_gesture else None
        target_after = pre_post.get(target_gesture, {}).get("after") if target_gesture else None
        summary = (
            f"OVERALL rolling accuracy (all gestures):  before={_fmt(run.get('post_drift_accuracy'))}"
            f"  ->  after={_fmt(run.get('post_recalibration_accuracy'))}\n"
            f"CALIBRATED TARGET gesture ({target_gesture or 'n/a'}) recovery:  "
            f"before={_fmt(target_before)}  ->  after={_fmt(target_after)}"
        )
        ax.text(
            0.01,
            1.13,
            summary,
            transform=ax.transAxes,
            fontsize=8,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round", facecolor="#f0f4f8", edgecolor="#888888"),
        )

    fig.suptitle(
        "Per-gesture accuracy before vs. after recalibration — synthetic_demo_data\n"
        "structured synthetic electrode-shift stress test, not clinical validation — "
        "overall rolling accuracy and the calibrated target gesture's own recovery are reported separately (see boxed summary per subject)",
        fontsize=11,
        fontweight="bold",
        y=0.998,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_cost_token_summary(results: dict, out_path: Path) -> None:
    rows = [
        [r["subject"], r["condition"], r["data_source"], r["n_windows"],
         r["estimated_tokens"], f"${r['estimated_cost_usd']:.4f}"]
        for r in results["runs"]
    ]
    col_labels = ["subject", "condition", "data source", "windows", "est. tokens", "est. cost (USD)"]

    fig, ax = plt.subplots(figsize=(11, 0.32 * len(rows) + 1.0))
    ax.axis("off")
    table = ax.table(cellText=rows, colLabels=col_labels, cellLoc="left", colLoc="left", loc="upper center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.4)
    table.auto_set_column_width(col=list(range(len(col_labels))))
    for (row, _col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold", color="white")
            cell.set_facecolor("#1f6feb")

    ax.set_title(
        "Token / cost summary — estimated real-API cost, not measured (MOCK path makes no API calls)",
        fontsize=12, fontweight="bold", pad=12,
    )
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def plot_baseline_comparison(baselines: dict, out_path: Path) -> None:
    rows = baselines["rows"]
    subjects = [r["subject"] for r in rows]
    knn = [r["knn_rms_accuracy"] or 0.0 for r in rows]
    mock = [r["myocopilot_mock_clean_accuracy"] or 0.0 for r in rows]
    sources = [r["data_source"] for r in rows]

    x = np.arange(len(subjects))
    width = 0.38
    fig, ax = plt.subplots(figsize=(2.2 * len(subjects) + 2, 5))
    ax.bar(x - width / 2, knn, width, label=f"k-NN on RMS (k={baselines['k_neighbors']})", color="#8250df")
    ax.bar(x + width / 2, mock, width, label="MyoCopilot MOCK path (clean)", color="#1f6feb")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n[{src}]" for s, src in zip(subjects, sources)], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("accuracy")
    ax.legend(fontsize=9)
    ax.set_title(
        "Classical baseline comparison — sanity check, not a superiority claim\n"
        f"({baselines['note']})",
        fontsize=10, fontweight="bold", wrap=True,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    results = _load_results()
    baselines = _load_baselines()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    plot_confusion_matrices(results, OUT_DIR / "confusion_matrices.png")
    plot_accuracy_vs_time(results, OUT_DIR / "accuracy_vs_time.png")
    plot_recalibration_before_after(results, OUT_DIR / "recalibration_before_after.png")
    plot_cost_token_summary(results, OUT_DIR / "cost_token_summary.png")

    if baselines is not None:
        plot_baseline_comparison(baselines, OUT_DIR / "baseline_comparison.png")
    else:
        print("eval/baseline_results.json not found — skipping baseline_comparison.png "
              "(run `python -m eval.baselines` first to include it)")

    print(f"Wrote figures to {OUT_DIR}/")


if __name__ == "__main__":
    main()
