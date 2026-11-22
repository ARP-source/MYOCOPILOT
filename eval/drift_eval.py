"""
eval.drift_eval: drift detectors on real NinaPro DB2 data (exercise B).

Same subjects, features, train/test split and simulated drift as
eval/real_eval.py. Compares:

- old: agent.drift_detector's accuracy rule (rolling accuracy over 20
  labeled windows under 0.60), fed by the app's prototype classifier. Needs
  the true label.
- new: agent.feature_drift.FeatureDriftDetector. Label free. Fit and
  calibrated on the clean training repetitions (1, 3, 4, 6) only, by cross
  fitting over those 4 repetitions, with a target false alarm rate of 2%.

Protocol
--------
- False alarms: stream the clean test repetitions (2 and 5) in recording
  order and report the share of windows where the alarm is on. Windows
  before the detector's buffers are full are left out (the first 199 for the
  new detector, the first 19 for the old one). Also reported as alarm
  episodes (off to on switches) per minute of streamed windows.
- Detection: for each of 13 injection points spread from 20% to 80% of the
  test stream, the windows before the point are clean and the windows from
  it on are drifted. The new detector runs over the whole stream (no reset
  at the injection point, since a real system would not know when drift
  starts). Drift counts as detected if the alarm is on at any window within
  30 s (600 windows) after the injection point. Delay is the median over
  detected injections, in seconds of streamed windows (50 ms per window).
  The old detector is restarted at the injection point, as in
  eval/real_eval.py, and its share of clean windows where the rule is
  already true is reported next to it.
- Drift types: rotate (forearm ring channels 1-8 moved by one electrode),
  gain x0.6, rotate and gain, and a milder gain x0.8. "none" runs the same
  protocol with no drift at all, so its "detection rate" is the chance that
  a false alarm lands inside the 30 s window. Read the others against it.

Also reports, on the same protocol, a few other label free signals for the
pattern part (global Mahalanobis distance and the prototype classifier's own
confidence), and the level threshold settings that were tried.

Never calls the network or any LLM.

Run with:
    python -m eval.drift_eval --data-dir data/ninapro_db2
Writes eval/drift_results.json and eval/report/real_drift_detector.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from multiprocessing import Pool
from pathlib import Path

os.environ["MOCK"] = "1"

import numpy as np

from eval.real_eval import (
    Subject, RING, STRIDE_MS, TRAIN_REPS, TEST_REPS, ROLLING_WINDOW,
    _window_features, settled_indices, build_prototype, proto_predict,
)
from agent.feature_drift import (
    FeatureDriftDetector, rolling_mean, persistence, calibrate_threshold,
    fit_stats, _transform,
)

OUT_JSON = Path("eval/drift_results.json")
FIG = Path("eval/report/real_drift_detector.png")
DRIFTS = {
    "rotate": (True, 1.0),
    "gain_0.6": (False, 0.6),
    "rotate_gain_0.6": (True, 0.6),
    "gain_0.8": (False, 0.8),
    "none": (False, 1.0),      # control: same protocol with no drift at all
}
INJECT_FRACTIONS = np.linspace(0.2, 0.8, 13)
HORIZON = 600                  # 30 s of windows
TARGET_FA = 0.02
OLD_ACC_THRESHOLD = 0.60
WIN_S = STRIDE_MS / 1000


def drifted(subj: Subject, rotate: bool, gain: float) -> np.ndarray:
    f = subj.filtered
    if rotate:
        f = f.copy()
        f[:, :RING] = np.roll(f[:, :RING], 1, axis=1)
    if gain != 1.0:
        f = f * gain
    if not rotate and gain == 1.0:
        return subj.X.copy()
    z = ((f - subj.mean) / subj.std).astype(np.float32)
    starts, rms, zc = _window_features(z, subj.fs)
    idx = subj.starts // int(round(STRIDE_MS * subj.fs / 1000))
    assert np.array_equal(starts[idx], subj.starts)
    return subj._vec(rms[idx], zc[idx])


def episodes_per_min(alarm: np.ndarray) -> float:
    a = alarm.astype(int)
    starts = int(np.sum(np.diff(np.concatenate([[0], a])) == 1))
    return round(starts / (len(a) * WIN_S / 60), 3)


def _round(x, n=4):
    return None if x is None else round(float(x), n)


# ---------------------------------------------------------------------------
# New detector
# ---------------------------------------------------------------------------

def eval_new(det: FeatureDriftDetector, Xc: np.ndarray, Xd: dict) -> dict:
    warm = det.level_window - 1
    r = det.run(Xc)
    out = {
        "pattern_threshold": _round(det.pattern_threshold),
        "level_threshold_log": _round(det.level_threshold),
        "clean_false_alarm_frac": _round(r["alarm"][warm:].mean()),
        "clean_false_alarm_frac_pattern": _round(r["pattern_alarm"][warm:].mean()),
        "clean_false_alarm_frac_level": _round(r["level_alarm"][warm:].mean()),
        "clean_alarm_episodes_per_min": episodes_per_min(r["alarm"][warm:]),
        "drift": {},
    }
    N = len(Xc)
    for name, X in Xd.items():
        dets, delays, by = [], [], {"pattern": 0, "level": 0}
        for fr in INJECT_FRACTIONS:
            p = int(N * fr)
            rr = det.run(np.vstack([Xc[:p], X[p:]]))
            hit = np.flatnonzero(rr["alarm"][p:p + HORIZON])
            dets.append(len(hit) > 0)
            if len(hit):
                delays.append(hit[0] * WIN_S)
                by["pattern" if rr["pattern_alarm"][p + hit[0]] else "level"] += 1
        out["drift"][name] = {
            "detection_rate": _round(np.mean(dets), 3),
            "median_delay_s": _round(np.median(delays), 2) if delays else None,
            "first_alarm_from": by,
        }
    return out


# ---------------------------------------------------------------------------
# Old detector (accuracy rule, needs labels)
# ---------------------------------------------------------------------------

def old_rule(correct: np.ndarray) -> np.ndarray:
    acc = rolling_mean(correct.astype(float), ROLLING_WINDOW)
    return np.where(np.isnan(acc), False, acc < OLD_ACC_THRESHOLD)


def eval_old(proto, Xc, yc, Xd) -> dict:
    pc = proto_predict(proto, Xc)
    rule_clean = old_rule(pc == yc)
    out = {
        "clean_accuracy": _round(np.mean(pc == yc)),
        "clean_rule_true_frac": _round(rule_clean[ROLLING_WINDOW - 1:].mean()),
        "clean_alarm_episodes_per_min": episodes_per_min(rule_clean[ROLLING_WINDOW - 1:]),
        "drift": {},
    }
    N = len(Xc)
    for name, X in Xd.items():
        pd_ = proto_predict(proto, X)
        dets, delays, already = [], [], []
        for fr in INJECT_FRACTIONS:
            p = int(N * fr)
            r = old_rule(pd_[p:] == yc[p:])     # restarted at the injection point
            hit = np.flatnonzero(r[:HORIZON])
            dets.append(len(hit) > 0)
            if len(hit):
                delays.append(hit[0] * WIN_S)
            already.append(bool(rule_clean[p - 1]))
        out["drift"][name] = {
            "drifted_accuracy": _round(np.mean(pd_ == yc)),
            "detection_rate": _round(np.mean(dets), 3),
            "median_delay_s": _round(np.median(delays), 2) if delays else None,
            "rule_already_true_on_clean_just_before_frac": _round(np.mean(already), 3),
        }
    return out


# ---------------------------------------------------------------------------
# Other label free signals for the pattern part, same calibration
# ---------------------------------------------------------------------------

def _global_maha(Xfit):
    n_ch = Xfit.shape[1] // 2
    Z = _transform(Xfit, n_ch)
    mu, sd = Z.mean(0), Z.std(0) + 1e-6
    Zs = (Z - mu) / sd
    prec = np.linalg.inv(np.cov(Zs.T) + 1e-3 * np.eye(Zs.shape[1]))

    def score(X):
        D = (_transform(X, n_ch) - mu) / sd - Zs.mean(0)
        return np.sqrt(np.einsum("nd,de,ne->n", D, prec, D))
    return score


def _proto_conf(subj, reps):
    s_idx = settled_indices(subj, reps)
    st = build_prototype(subj.X[s_idx], subj.y[s_idx])
    mean, std = np.array(st.norm_mean), np.array(st.norm_std)
    names, ex = [], []
    for g, pr in st.profiles.items():
        for e in pr.examples:
            names.append(g)
            ex.append(e)
    names, ex = np.array(names), np.array(ex)
    classes = np.unique(names)

    def score(X):
        Xn = (X - mean) / std
        d = np.sqrt(((Xn[:, None, :] - ex[None]) ** 2).sum(-1))
        per = np.stack([d[:, names == c].min(1) for c in classes], 1)
        per.sort(1)
        dn, ds = per[:, 0], per[:, 1]
        conf = np.clip((ds - dn) / (ds + dn + 1e-6) * 2, 0.05, 0.98)
        unfamiliar = (dn / max(st.typical_distance, 1e-6) > 3.0) | (conf < 0.15)
        conf = np.where(unfamiliar, np.minimum(conf, 0.3), conf)
        return 1.0 - conf
    return score


def _class_maha(Xfit, yfit):
    from agent.feature_drift import window_scores
    st = fit_stats(Xfit, yfit)
    return lambda X: window_scores(st, X)[0]


def eval_pattern_signals(subj, tr, te, Xd, det: FeatureDriftDetector) -> dict:
    m, k, n = det.pattern_window, det.persist_k, det.persist_n
    makers = {
        "class_mahalanobis (chosen)": lambda f: _class_maha(subj.X[f], subj.y[f]),
        "global_mahalanobis": lambda f: _global_maha(subj.X[f]),
        "prototype_confidence": lambda f: _proto_conf(subj, sorted(set(subj.rep[f]))),
    }
    out = {}
    N = int(te.sum())
    for name, make in makers.items():
        cal = []
        for r in TRAIN_REPS:
            f = tr & (subj.rep != r)
            cal.append(rolling_mean(make(f)(subj.X[subj.rep == r]), m))
        tau = calibrate_threshold(cal, k, n, TARGET_FA / 2)
        score = make(tr)
        sc = score(subj.X[te])

        def alarm(s):
            rm = rolling_mean(s, m)
            return persistence(np.where(np.isnan(rm), -np.inf, rm) > tau, k, n)
        row = {"clean_false_alarm_frac": _round(alarm(sc)[m - 1:].mean())}
        for dname in ("rotate", "gain_0.6"):
            sd_ = score(Xd[dname])
            dets = []
            for fr in INJECT_FRACTIONS:
                p = int(N * fr)
                dets.append(bool(alarm(np.concatenate([sc[:p], sd_[p:]]))[p:p + HORIZON].any()))
            row[f"{dname}_detection_rate"] = _round(np.mean(dets), 3)
        out[name] = row
    return out


# ---------------------------------------------------------------------------
# Level threshold settings tried
# ---------------------------------------------------------------------------

LEVEL_SETTINGS = [(200, 1.0), (200, 1.25), (400, 1.0), (400, 1.25)]


def eval_level_settings(Xtr, ytr, gtr, Xc, Xd) -> dict:
    out = {}
    for lw, margin in LEVEL_SETTINGS:
        det = FeatureDriftDetector(level_window=lw, level_margin=margin,
                                   target_false_alarm=TARGET_FA).fit(Xtr, ytr, gtr)
        r = eval_new(det, Xc, {k: Xd[k] for k in ("gain_0.6", "gain_0.8")})
        out[f"level_window_{lw}_margin_{margin}"] = {
            "clean_false_alarm_frac": r["clean_false_alarm_frac"],
            "gain_0.6_detection_rate": r["drift"]["gain_0.6"]["detection_rate"],
            "gain_0.8_detection_rate": r["drift"]["gain_0.8"]["detection_rate"],
        }
    return out


# ---------------------------------------------------------------------------

def run_subject(path: str) -> dict:
    subj = Subject(Path(path))
    tr = np.isin(subj.rep, TRAIN_REPS)
    te = np.isin(subj.rep, TEST_REPS)
    Xd_all = {k: drifted(subj, *v) for k, v in DRIFTS.items()}
    Xd = {k: v[te] for k, v in Xd_all.items()}
    Xc, yc = subj.X[te], subj.y[te]

    det = FeatureDriftDetector(target_false_alarm=TARGET_FA).fit(subj.X[tr], subj.y[tr], subj.rep[tr])
    s_idx = settled_indices(subj, TRAIN_REPS)
    proto = build_prototype(subj.X[s_idx], subj.y[s_idx])

    row = {
        "subject": subj.name,
        "n_test_windows": int(te.sum()),
        "new": eval_new(det, Xc, Xd),
        "old": eval_old(proto, Xc, yc, Xd),
        "pattern_signals": eval_pattern_signals(subj, tr, te, Xd, det),
        "level_settings": eval_level_settings(subj.X[tr], subj.y[tr], subj.rep[tr], Xc, Xd),
    }
    print(row["subject"], json.dumps(row["new"]), flush=True)
    return row


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def summarise(rows):
    s = {"new": {}, "old": {}, "pattern_signals": {}, "level_settings": {}}
    for key in ("clean_false_alarm_frac", "clean_false_alarm_frac_pattern",
                "clean_false_alarm_frac_level", "clean_alarm_episodes_per_min"):
        s["new"][key] = _mean([r["new"][key] for r in rows])
    s["new"]["drift"] = {
        d: {
            "detection_rate": _mean([r["new"]["drift"][d]["detection_rate"] for r in rows]),
            "median_delay_s_per_subject": [r["new"]["drift"][d]["median_delay_s"] for r in rows],
            "mean_of_subject_median_delays_s": _mean([r["new"]["drift"][d]["median_delay_s"] for r in rows]),
        }
        for d in DRIFTS
    }
    for key in ("clean_accuracy", "clean_rule_true_frac", "clean_alarm_episodes_per_min"):
        s["old"][key] = _mean([r["old"][key] for r in rows])
    s["old"]["drift"] = {
        d: {
            "detection_rate": _mean([r["old"]["drift"][d]["detection_rate"] for r in rows]),
            "mean_of_subject_median_delays_s": _mean([r["old"]["drift"][d]["median_delay_s"] for r in rows]),
            "rule_already_true_on_clean_just_before_frac": _mean(
                [r["old"]["drift"][d]["rule_already_true_on_clean_just_before_frac"] for r in rows]),
        }
        for d in DRIFTS
    }
    for name in rows[0]["pattern_signals"]:
        s["pattern_signals"][name] = {
            k: _mean([r["pattern_signals"][name][k] for r in rows]) for k in rows[0]["pattern_signals"][name]
        }
    for name in rows[0]["level_settings"]:
        s["level_settings"][name] = {
            k: _mean([r["level_settings"][name][k] for r in rows]) for k in rows[0]["level_settings"][name]
        }
    return s


def figure(rows) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG.parent.mkdir(parents=True, exist_ok=True)
    subs = [r["subject"] for r in rows]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(subs))
    w = 0.38
    old = [r["old"]["clean_rule_true_frac"] for r in rows]
    new = [r["new"]["clean_false_alarm_frac"] for r in rows]
    a1.bar(x - w / 2, old, w, label="Old: accuracy under 0.60 (needs labels)")
    a1.bar(x + w / 2, new, w, label="New: feature shift (no labels)")
    for xi, v in zip(x, new):
        a1.text(xi + w / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=7)
    a1.set_xticks(x, subs)
    a1.set_ylim(0, 1.05)
    a1.set_ylabel("Share of clean test windows with alarm on")
    a1.set_title("False alarms on clean data")
    a1.legend(fontsize=7, loc="upper left")

    labels = {"rotate": "Rotate", "gain_0.6": "Gain x0.6", "rotate_gain_0.6": "Both",
              "gain_0.8": "Gain x0.8", "none": "No drift\n(control)"}
    xs = np.arange(len(DRIFTS))
    means = [np.mean([r["new"]["drift"][d]["detection_rate"] for r in rows]) for d in DRIFTS]
    a2.bar(xs, means, 0.6, color="C1", alpha=0.6, label="New detector, mean")
    for i, d in enumerate(DRIFTS):
        pts = [r["new"]["drift"][d]["detection_rate"] for r in rows]
        a2.scatter([i] * len(pts), pts, s=12, color="k", zorder=3)
        delays = [r["new"]["drift"][d]["median_delay_s"] for r in rows if r["new"]["drift"][d]["median_delay_s"] is not None]
        if delays:
            a2.text(i, 1.06, f"{np.median(delays):.1f} s", ha="center", fontsize=8)
    a2.set_xticks(xs, [labels[d] for d in DRIFTS])
    a2.set_ylim(0, 1.15)
    a2.set_ylabel("Share of injections caught within 30 s")
    a2.set_title("New detector: detection (dots = subjects, text = median delay)", fontsize=9)
    fig.suptitle(f"NinaPro DB2 exercise B, {len(rows)} subjects, simulated drift", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG, dpi=130)
    plt.close(fig)
    return str(FIG)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ninapro_db2")
    ap.add_argument("--jobs", type=int, default=5)
    args = ap.parse_args()
    files = sorted(Path(args.data_dir).glob("S*_E1_A1.mat"), key=lambda p: int(p.stem.split("_")[0][1:]))
    if not files:
        sys.exit(f"No S*_E1_A1.mat files in {args.data_dir}")
    with Pool(max(1, args.jobs)) as pool:
        rows = pool.map(run_subject, [str(f) for f in files])
    d = FeatureDriftDetector()
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "NinaPro DB2, exercise B (S*_E1_A1.mat), same windows and split as eval/real_eval.py",
        "subjects": [r["subject"] for r in rows],
        "new_detector": {
            "class": "agent.feature_drift.FeatureDriftDetector",
            "fit_and_calibration_data": f"clean training reps {TRAIN_REPS} only, cross fit over those reps",
            "target_false_alarm": TARGET_FA,
            "pattern_window": d.pattern_window,
            "persistence": f"{d.persist_k} of {d.persist_n} windows",
            "level_window": d.level_window,
            "level_margin": d.level_margin,
            "uses_true_label_after_enrollment": False,
        },
        "old_detector": "agent.drift_detector accuracy rule: rolling accuracy over 20 labeled windows < 0.60, prototype classifier (4 examples per movement)",
        "protocol": {
            "false_alarm": "share of clean test windows (reps 2, 5, recording order) with alarm on, after buffers are full",
            "injection_points": [round(float(f), 3) for f in INJECT_FRACTIONS],
            "detection_horizon_s": HORIZON * WIN_S,
            "new_detector_stream": "runs from the start of the test stream, never reset at the injection point",
            "old_detector_stream": "restarted at the injection point, as in eval/real_eval.py",
            "drifts": {k: {"rotate_ring_by_one": v[0], "gain": v[1]} for k, v in DRIFTS.items()},
            "level_settings_note": (
                "Four level settings were run and the default (window 200, margin 1.25) was picked "
                "after seeing their clean test false alarm rates, so the new false alarm number is "
                "slightly optimistic. All four are reported."
            ),
        },
        "summary_mean_over_subjects": summarise(rows),
        "per_subject": rows,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    fig = figure(rows)
    print(json.dumps(result["summary_mean_over_subjects"], indent=1))
    print("wrote", OUT_JSON, fig)


if __name__ == "__main__":
    main()
