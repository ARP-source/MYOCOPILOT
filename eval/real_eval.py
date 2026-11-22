"""
eval.real_eval: evaluation on real NinaPro DB2 recordings (exercise B).

What it does
------------
For each subject file S<n>_E1_A1.mat (exercise B: 17 movements, 6 repetitions):

1. Preprocess with myosignal.preprocess (20-450 Hz bandpass, per-channel
   z-score from the first 30 s), cut 250 ms windows with a 50 ms stride and
   compute the same RMS and zero crossing features the app uses.
   Only windows whose samples all carry one movement label and one
   repetition are kept. Rest (label 0) is left out, so this is a 17 class
   problem (chance is about 0.06).
2. Standard NinaPro split: train on repetitions 1, 3, 4, 6, test on 2 and 5.
3. Classifiers, all on the same 24 dim feature vector
   (12 RMS + 12 zero crossings / 50, as in agent/mock_classifier.py):
   - prototype: the app's nearest prototype classifier
     (agent.mock_classifier._SubjectState). Its profile is built the way the
     app enrolls during a live session: one settled window (the 4th window)
     per repetition, so 4 stored examples per movement.
   - lda_4: scikit-learn LDA trained on exactly those same 4 windows per
     movement (same data as the prototype classifier).
   - lda_all: LDA trained on every training window.
   - nn1_all: nearest neighbour over every training window (the prototype
     rule with no cap on stored examples), for reference.
4. Simulated drift on the test repetitions, applied to the bandpassed signal
   before the frozen z-score (so normalisation stays calibrated to the clean
   start of the session, like a real device would be):
   - rotate: the 8 electrodes in the forearm ring (channels 1-8) rotated by
     one position. Channels 9-12 are placed on specific muscles and are left
     alone.
   - gain: every channel scaled by 0.6 (lower skin contact).
   - rotate_gain: both.
5. Recalibration under drift with k = 3, 5, 10 examples per movement, taken
   from the drifted TRAINING repetitions (never the test ones), spread evenly
   over them:
   - all movements recalibrated (prototype via enroll_confirmed, LDA refit on
     the k drifted examples per movement);
   - targeted, the way the app does it: stream the test windows, start the
     DriftDetector at the drift injection point, recalibrate only the
     movement check_drift() names, then keep streaming.
6. Early vs late repetitions: train on reps 1-3, test on 4-6. DB2 has only
   one session per subject, so this is not a cross-session test.

Never calls the network or any LLM.

Run with:
    python -m eval.real_eval --data-dir data/ninapro_db2
Writes eval/real_results.json and eval/report/real_*.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ["MOCK"] = "1"  # never import or call the LLM classifier

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier

from myosignal.loaders import load_ninapro
from myosignal.preprocess import preprocess
from myosignal.windows import WINDOW_MS, STRIDE_MS, compute_rms, compute_zero_crossings
from agent.mock_classifier import _SubjectState, ZC_NORM_SCALE, ENROLL_SETTLE_WINDOWS, PROTO_CAP
from agent.drift_detector import DriftDetector, check_drift

TRAIN_REPS = (1, 3, 4, 6)
TEST_REPS = (2, 5)
EARLY_REPS = (1, 2, 3)
LATE_REPS = (4, 5, 6)
RING = 8                 # channels 1-8 are the equally spaced forearm ring
DRIFT_GAIN = 0.6
RECAL_K = (3, 5, 10)
DRIFT_FRACTION = 0.35    # same injection point as eval/run.py
ROLLING_WINDOW = 20      # same as eval/run.py / DriftDetector default
OUT_JSON = Path("eval/real_results.json")
FIG_DIR = Path("eval/report")

LABEL_MAP = {i: f"B{i}" for i in range(1, 18)}
LABEL_MAP[0] = ""


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def _window_features(emg: np.ndarray, fs: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised RMS and zero crossings for every 250 ms / 50 ms window.

    Same definitions as myosignal.windows.compute_rms / compute_zero_crossings
    (checked against them in _check_features)."""
    win = int(round(WINDOW_MS * fs / 1000))
    hop = int(round(STRIDE_MS * fs / 1000))
    n = emg.shape[0]
    starts = np.arange(0, n - win + 1, hop)
    csq = np.vstack([np.zeros((1, emg.shape[1])), np.cumsum(emg.astype(np.float64) ** 2, axis=0)])
    rms = np.sqrt((csq[starts + win] - csq[starts]) / win).astype(np.float32)
    s = np.sign(emg)
    # carry the previous sign through exact zeros (rare after filtering)
    if np.any(s == 0):
        for ch in range(s.shape[1]):
            col = s[:, ch]
            idx = np.where(col != 0, np.arange(len(col)), 0)
            np.maximum.accumulate(idx, out=idx)
            col = col[idx]
            col[col == 0] = 1
            s[:, ch] = col
    flips = (np.abs(np.diff(s, axis=0)) > 1).astype(np.int32)
    cfl = np.vstack([np.zeros((1, emg.shape[1]), dtype=np.int64), np.cumsum(flips, axis=0)])
    zc = (cfl[starts + win - 1] - cfl[starts]).astype(np.int32)
    return starts, rms, zc


def _check_features(emg: np.ndarray, fs: int, starts, rms, zc, n_check: int = 40) -> None:
    win = int(round(WINDOW_MS * fs / 1000))
    rng = np.random.default_rng(0)
    for i in rng.choice(len(starts), size=min(n_check, len(starts)), replace=False):
        w = emg[starts[i]:starts[i] + win]
        assert np.allclose(compute_rms(w), rms[i], rtol=1e-4, atol=1e-5)
        assert np.array_equal(compute_zero_crossings(w.copy()), zc[i])


class Subject:
    def __init__(self, path: Path):
        raw = load_ninapro(path, label_map=LABEL_MAP)
        self.name = path.stem.split("_")[0]
        self.fs = raw.fs
        z, mean, std = preprocess(raw.emg, raw.fs, notch=False, baseline_s=30.0)
        self.mean, self.std = mean, std
        self.filtered = z * std + mean          # bandpassed, before z-score
        self.labels = raw.labels
        self.reps = raw.reps
        win = int(round(WINDOW_MS * raw.fs / 1000))
        starts, rms, zc = _window_features(z, raw.fs)
        _check_features(z, raw.fs, starts, rms, zc)
        # keep windows with a single movement label and a single repetition
        lab_ok = np.array([self.labels[s] == self.labels[s + win - 1] for s in starts])
        cl = np.concatenate([[0], np.cumsum(np.diff(self.labels.astype(np.int64)) != 0)])
        cr = np.concatenate([[0], np.cumsum(np.diff(self.reps.astype(np.int64)) != 0)])
        pure = (cl[starts + win - 1] == cl[starts]) & (cr[starts + win - 1] == cr[starts]) & lab_ok
        keep = pure & (self.labels[starts] > 0)
        self.starts = starts[keep]
        self.y = self.labels[self.starts].astype(int)
        self.rep = self.reps[self.starts].astype(int)
        self.X = self._vec(rms[keep], zc[keep])

    @staticmethod
    def _vec(rms, zc):
        return np.hstack([rms, zc / ZC_NORM_SCALE]).astype(np.float64)

    def drifted_features(self, mode: str) -> np.ndarray:
        """Features of the kept windows after applying *mode* drift."""
        f = self.filtered
        if mode in ("rotate", "rotate_gain"):
            f = f.copy()
            f[:, :RING] = np.roll(f[:, :RING], 1, axis=1)
        if mode in ("gain", "rotate_gain"):
            f = f * DRIFT_GAIN
        z = ((f - self.mean) / self.std).astype(np.float32)
        starts, rms, zc = _window_features(z, self.fs)
        idx = self.starts // int(round(STRIDE_MS * self.fs / 1000))
        assert np.array_equal(starts[idx], self.starts)
        return self._vec(rms[idx], zc[idx])


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------

def settled_indices(subj: Subject, reps) -> np.ndarray:
    """One window per (movement, repetition) run: the ENROLL_SETTLE_WINDOWS-th
    window after the run starts, as agent.mock_classifier enrolls live."""
    out = []
    for g in np.unique(subj.y):
        for r in reps:
            idx = np.flatnonzero((subj.y == g) & (subj.rep == r))
            if len(idx) > ENROLL_SETTLE_WINDOWS:
                out.append(idx[ENROLL_SETTLE_WINDOWS])
    return np.array(sorted(out))


def build_prototype(X: np.ndarray, y: np.ndarray) -> _SubjectState:
    st = _SubjectState()
    st._bootstrap_pool = [(f"B{g}", list(map(float, v))) for v, g in zip(X, y)]
    st._freeze_normalization()
    return st


def proto_predict(st: _SubjectState, X: np.ndarray) -> np.ndarray:
    """Same rule as _SubjectState.predict (nearest stored example after the
    frozen z-score), vectorised."""
    mean = np.array(st.norm_mean)
    std = np.array(st.norm_std)
    Xn = (X - mean) / std
    names, ex = [], []
    for g, p in st.profiles.items():
        for e in p.examples:
            names.append(int(g[1:]))
            ex.append(e)
    ex = np.array(ex)
    d = ((Xn[:, None, :] - ex[None, :, :]) ** 2).sum(-1)
    return np.array(names)[np.argmin(d, axis=1)]


def _check_proto(st, X):
    vec = proto_predict(st, X[:30])
    ref = [int(st.predict(list(map(float, x)))["gesture"][1:]) for x in X[:30]]
    assert list(vec) == ref


def lda():
    return LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")


def acc(a, b) -> float:
    return round(float(np.mean(np.asarray(a) == np.asarray(b))), 4)


def even_pick(idx: np.ndarray, k: int) -> np.ndarray:
    if len(idx) <= k:
        return idx
    return idx[np.round(np.linspace(0, len(idx) - 1, k)).astype(int)]


# ---------------------------------------------------------------------------
# Experiments
# ---------------------------------------------------------------------------

def run_subject(subj: Subject) -> dict:
    tr = np.isin(subj.rep, TRAIN_REPS)
    te = np.isin(subj.rep, TEST_REPS)
    Xtr, ytr, Xte, yte = subj.X[tr], subj.y[tr], subj.X[te], subj.y[te]
    s_idx = settled_indices(subj, TRAIN_REPS)
    Xs, ys = subj.X[s_idx], subj.y[s_idx]

    proto = build_prototype(Xs, ys)
    _check_proto(proto, Xte)
    lda4 = lda().fit(Xs, ys)
    lda_all = lda().fit(Xtr, ytr)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-9
    nn1 = KNeighborsClassifier(1).fit((Xtr - mu) / sd, ytr)

    res = {
        "subject": subj.name,
        "n_train_windows": int(tr.sum()),
        "n_test_windows": int(te.sum()),
        "n_prototype_examples_per_class": int(len(ys) / len(np.unique(ys))),
        "clean": {
            "prototype": acc(proto_predict(proto, Xte), yte),
            "lda_4": acc(lda4.predict(Xte), yte),
            "lda_all": acc(lda_all.predict(Xte), yte),
            "nn1_all": acc(nn1.predict((Xte - mu) / sd), yte),
        },
        "drift": {},
        "recal_all": {},
        "targeted": {},
    }

    for mode in ("rotate", "gain", "rotate_gain"):
        Xd = subj.drifted_features(mode)
        Xdte = Xd[te]
        res["drift"][mode] = {
            "prototype": acc(proto_predict(proto, Xdte), yte),
            "lda_4": acc(lda4.predict(Xdte), yte),
            "lda_all": acc(lda_all.predict(Xdte), yte),
        }
        # recalibrate every movement with k drifted examples from training reps
        res["recal_all"][mode] = {}
        for k in RECAL_K:
            pick = np.concatenate([
                even_pick(np.flatnonzero(tr & (subj.y == g)), k) for g in np.unique(subj.y)
            ])
            st = build_prototype(Xs, ys)
            for g in np.unique(subj.y):
                gi = pick[subj.y[pick] == g]
                st.enroll_confirmed(f"B{g}", [list(map(float, v)) for v in Xd[gi]])
            lda_k = lda().fit(Xd[pick], subj.y[pick])
            res["recal_all"][mode][str(k)] = {
                "prototype": acc(proto_predict(st, Xdte), yte),
                "prototype_examples_stored": min(k, PROTO_CAP),
                "lda_refit": acc(lda_k.predict(Xdte), yte),
            }
        res["targeted"][mode] = {str(k): targeted_stream(subj, Xs, ys, Xd, tr, te, k) for k in RECAL_K}

    # early vs late repetitions (same session)
    e = np.isin(subj.rep, EARLY_REPS)
    l = np.isin(subj.rep, LATE_REPS)
    se = settled_indices(subj, EARLY_REPS)
    pe = build_prototype(subj.X[se], subj.y[se])
    res["early_vs_late"] = {
        "prototype": acc(proto_predict(pe, subj.X[l]), subj.y[l]),
        "lda_3": acc(lda().fit(subj.X[se], subj.y[se]).predict(subj.X[l]), subj.y[l]),
        "lda_all": acc(lda().fit(subj.X[e], subj.y[e]).predict(subj.X[l]), subj.y[l]),
    }
    return res


def targeted_stream(subj, Xs, ys, Xd, tr, te, k) -> dict:
    """Stream the test windows in recording order, inject drift at 35%, run
    the app's DriftDetector/check_drift from the injection point, and
    recalibrate only the flagged movement with k drifted training examples."""
    st = build_prototype(Xs, ys)
    te_idx = np.flatnonzero(te)
    n = len(te_idx)
    inj = int(n * DRIFT_FRACTION)
    clean_pred = proto_predict(st, subj.X[te_idx])
    drift_pred_static = proto_predict(st, Xd[te_idx])

    # How often the detector's accuracy rule already fires on CLEAN test data.
    det_clean = DriftDetector(window_size=ROLLING_WINDOW)
    fires = 0
    for i in range(inj):
        det_clean.update(f"B{subj.y[te_idx[i]]}", {"gesture": f"B{clean_pred[i]}", "confidence": 0.5})
        if len(det_clean._recent) >= ROLLING_WINDOW and det_clean.rolling_accuracy() < det_clean.accuracy_threshold:
            fires += 1
    n_full = max(1, inj - ROLLING_WINDOW + 1)

    det = DriftDetector(window_size=ROLLING_WINDOW)
    preds = list(clean_pred[:inj])
    detected_at = None
    pair = None
    i = inj
    while i < n:
        p = drift_pred_static[i] if detected_at is None else None
        if p is None:
            break
        preds.append(p)
        det.update(f"B{subj.y[te_idx[i]]}", {"gesture": f"B{p}", "confidence": 0.5})
        hit = check_drift(det)
        if hit is not None:
            detected_at = i
            pair = hit
        i += 1
    recal_from = i
    if pair is not None:
        g = int(pair[0][1:])
        gi = even_pick(np.flatnonzero(tr & (subj.y == g)), k)
        st.enroll_confirmed(pair[0], [list(map(float, v)) for v in Xd[gi]])
    after = proto_predict(st, Xd[te_idx[recal_from:]]) if recal_from < n else np.array([])
    y_after = subj.y[te_idx[recal_from:]]
    out = {
        "clean_windows_with_accuracy_rule_true_frac": round(fires / n_full, 4),
        "drift_injected_at_test_window": inj,
        "detected": pair is not None,
        "detection_latency_windows": None if detected_at is None else detected_at - inj,
        "detection_latency_s": None if detected_at is None else round((detected_at - inj) * STRIDE_MS / 1000, 2),
        "flagged_pair": None if pair is None else list(pair),
        "post_drift_before_recal": acc(drift_pred_static[inj:recal_from], subj.y[te_idx[inj:recal_from]]) if recal_from > inj else None,
        "post_recal_overall": acc(after, y_after) if len(after) else None,
        "no_recal_same_span": acc(drift_pred_static[recal_from:], y_after) if len(after) else None,
    }
    if pair is not None and len(after):
        g = int(pair[0][1:])
        m = y_after == g
        out["flagged_movement_after"] = acc(after[m], y_after[m]) if m.any() else None
        out["flagged_movement_no_recal"] = acc(drift_pred_static[recal_from:][m], y_after[m]) if m.any() else None
    return out


# ---------------------------------------------------------------------------
# Summary, figures
# ---------------------------------------------------------------------------

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return round(float(np.mean(xs)), 4) if xs else None


def summarise(rows: list[dict]) -> dict:
    s = {"clean": {}, "drift": {}, "recal_all": {}, "early_vs_late": {}, "targeted": {}}
    for m in rows[0]["clean"]:
        s["clean"][m] = _mean([r["clean"][m] for r in rows])
    for mode in rows[0]["drift"]:
        s["drift"][mode] = {m: _mean([r["drift"][mode][m] for r in rows]) for m in rows[0]["drift"][mode]}
        s["recal_all"][mode] = {
            k: {m: _mean([r["recal_all"][mode][k][m] for r in rows]) for m in ("prototype", "lda_refit")}
            for k in rows[0]["recal_all"][mode]
        }
        s["targeted"][mode] = {
            k: {
                "detected": sum(r["targeted"][mode][k]["detected"] for r in rows),
                "latency_s": [r["targeted"][mode][k]["detection_latency_s"] for r in rows],
                "post_drift_before_recal": _mean([r["targeted"][mode][k]["post_drift_before_recal"] for r in rows]),
                "post_recal_overall": _mean([r["targeted"][mode][k]["post_recal_overall"] for r in rows]),
                "no_recal_same_span": _mean([r["targeted"][mode][k]["no_recal_same_span"] for r in rows]),
                "flagged_movement_after": _mean([r["targeted"][mode][k].get("flagged_movement_after") for r in rows]),
                "flagged_movement_no_recal": _mean([r["targeted"][mode][k].get("flagged_movement_no_recal") for r in rows]),
            }
            for k in rows[0]["targeted"][mode]
        }
    s["clean_accuracy_rule_true_frac"] = _mean(
        [r["targeted"]["rotate"]["3"]["clean_windows_with_accuracy_rule_true_frac"] for r in rows]
    )
    for m in rows[0]["early_vs_late"]:
        s["early_vs_late"][m] = _mean([r["early_vs_late"][m] for r in rows])
    return s


def figures(rows: list[dict]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    subs = [r["subject"] for r in rows]

    # 1. clean vs drift accuracy per classifier
    fig, ax = plt.subplots(figsize=(8, 4.2))
    conds = [("clean", "Clean"), ("rotate", "Rotate 1"), ("gain", "Gain 0.6"), ("rotate_gain", "Both")]
    models = [("prototype", "Prototype (4/class)"), ("lda_4", "LDA (4/class)"), ("lda_all", "LDA (all train)")]
    w = 0.26
    for j, (m, lab) in enumerate(models):
        vals = []
        for c, _ in conds:
            vals.append(np.mean([r["clean"][m] if c == "clean" else r["drift"][c][m] for r in rows]))
        x = np.arange(len(conds)) + (j - 1) * w
        ax.bar(x, vals, w, label=lab)
        for c_i, (c, _) in enumerate(conds):
            pts = [r["clean"][m] if c == "clean" else r["drift"][c][m] for r in rows]
            ax.scatter([x[c_i]] * len(pts), pts, s=8, color="k", zorder=3)
    ax.axhline(1 / 17, color="grey", ls="--", lw=1, label="Chance (1/17)")
    ax.set_xticks(np.arange(len(conds)), [c[1] for c in conds])
    ax.set_ylabel("Test accuracy (reps 2, 5)")
    ax.set_ylim(0, 1)
    ax.set_title(f"NinaPro DB2 exercise B, {len(rows)} subjects (dots = subjects)")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = FIG_DIR / "real_clean_vs_drift.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(str(p))

    # 2. recalibration curve under rotate + gain drift
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    ks = [0] + list(RECAL_K)
    for m, key0, lab, col in (("prototype", "prototype", "Prototype", "C0"), ("lda_refit", "lda_4", "LDA", "C1")):
        for mode, ls in (("rotate", "-"), ("rotate_gain", "--")):
            ys = [np.mean([r["drift"][mode][key0] for r in rows])]
            ys += [np.mean([r["recal_all"][mode][str(k)][m] for r in rows]) for k in RECAL_K]
            ax.plot(ks, ys, ls, color=col, marker="o", label=f"{lab}, {'rotate' if mode == 'rotate' else 'rotate + gain'}")
    ax.axhline(np.mean([r["clean"]["prototype"] for r in rows]), color="C0", lw=0.8, ls=":", label="Prototype (4/class), clean")
    ax.axhline(np.mean([r["clean"]["lda_4"] for r in rows]), color="C1", lw=0.8, ls=":", label="LDA (4/class), clean")
    ax.axhline(np.mean([r["clean"]["lda_all"] for r in rows]), color="grey", lw=0.8, ls=":", label="LDA (all train), clean")
    ax.set_xticks(ks)
    ax.set_xlabel("Recalibration examples per movement (0 = none)")
    ax.set_ylabel("Mean test accuracy under drift")
    ax.set_ylim(0, 1)
    ax.set_title("Recalibrating all movements after simulated drift")
    ax.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    p = FIG_DIR / "real_recalibration.png"
    fig.savefig(p, dpi=130)
    plt.close(fig)
    out.append(str(p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/ninapro_db2")
    args = ap.parse_args()
    files = sorted(Path(args.data_dir).glob("S*_E1_A1.mat"), key=lambda p: int(p.stem.split("_")[0][1:]))
    if not files:
        sys.exit(f"No S*_E1_A1.mat files in {args.data_dir}")
    rows = []
    for f in files:
        subj = Subject(f)
        r = run_subject(subj)
        rows.append(r)
        print(r["subject"], r["clean"], {m: r["drift"][m]["prototype"] for m in r["drift"]}, flush=True)
    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "dataset": "NinaPro DB2, exercise B (S*_E1_A1.mat), restimulus/rerepetition labels",
        "data_source_url": "https://doi.org/10.5061/dryad.1k84r",
        "subjects": [r["subject"] for r in rows],
        "classes": "17 movements, rest excluded",
        "split": {"train_reps": TRAIN_REPS, "test_reps": TEST_REPS},
        "windows": {"length_ms": WINDOW_MS, "stride_ms": STRIDE_MS, "kept": "single label and single repetition only"},
        "features": "per channel RMS (12) + zero crossings / 50 (12), after 20-450 Hz bandpass and z-score from first 30 s",
        "drift_modes": {
            "rotate": "forearm ring channels 1-8 rotated by one electrode position; channels 9-12 unchanged",
            "gain": f"all channels x{DRIFT_GAIN}",
            "rotate_gain": "both",
        },
        "recalibration_note": (
            f"k examples per movement from drifted training reps, spread evenly. The prototype "
            f"classifier stores at most PROTO_CAP={PROTO_CAP} examples, so k=10 keeps 8."
        ),
        "summary_mean_over_subjects": summarise(rows),
        "per_subject": rows,
    }
    OUT_JSON.write_text(json.dumps(result, indent=2))
    figs = figures(rows)
    print(json.dumps(result["summary_mean_over_subjects"], indent=1))
    print("wrote", OUT_JSON, figs)


if __name__ == "__main__":
    main()
