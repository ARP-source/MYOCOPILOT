"""
agent.feature_drift: a drift detector that does not need the true label.

Why this exists
---------------
agent.drift_detector.DriftDetector fires when rolling accuracy over 20
labeled windows drops under 0.60. On real NinaPro DB2 data the default
classifier only gets about 0.29 on clean data, so that rule is true on most
clean windows. It also needs the true gesture, which a real user does not
provide while using the hand.

This detector only looks at the features of each window and compares them
with statistics from a clean enrollment period. It watches two things:

pattern
    Per window, the Mahalanobis distance from the window's feature vector
    (log RMS per channel plus zero crossings per channel) to the nearest
    enrolled gesture mean, with a covariance pooled over gestures. When an
    electrode ring rotates, the channel pattern stops looking like any
    enrolled gesture and this distance jumps. The per-window distance is
    averaged over the last `pattern_window` windows, and the pattern alarm is
    on when that average is above its threshold in at least `persist_k` of
    the last `persist_n` windows.

level
    Per window, the mean log RMS over channels minus what a small ridge
    regression predicts from the zero crossing pattern alone. Zero crossings
    do not change when the signal is scaled, RMS does, so a uniform gain
    change (worse skin contact, sweat) shows up as a shift in this residual.
    The residual is averaged over the last `level_window` windows (10 s by
    default, so it averages over several gestures) and the level alarm is on
    when the absolute average is above its threshold.

Labels are used only during enrollment, to estimate gesture means, the same
way the classifier needs labeled enrollment windows. After that the detector
never sees a label.

Thresholds
----------
Both thresholds come from clean enrollment data only, by cross fitting:
enrollment windows are split into groups (repetitions), the statistics are
fit on all groups but one, and the held out group is streamed through the
detector.
- pattern: the smallest threshold whose alarm rate (with persistence) on the
  held out groups is at most target_false_alarm / 2.
- level: level_margin times the largest absolute rolling residual seen on the
  held out groups, but never below min_level_threshold (0.15 in log units,
  about a 14% amplitude change), which only matters for very regular data
  like the synthetic demo. The 10 s average leaves too few independent
  values in a few minutes of enrollment for a quantile to mean much, so it
  uses the max plus a margin instead. If no held out group is as long as
  level_window, the level check stays off.

Numbers on real data are in eval/drift_results.json (python -m eval.drift_eval).
"""

from __future__ import annotations

from collections import deque
from typing import Optional, Sequence

import numpy as np

ZC_NORM_SCALE = 50.0      # same scaling as agent.mock_classifier
LOG_EPS = 1e-3
STRIDE_S = 0.05


# ---------------------------------------------------------------------------
# Small helpers (also used by eval/drift_eval.py)
# ---------------------------------------------------------------------------

def envelope_vector(envelope: dict) -> list[float]:
    """RMS per channel followed by zero crossings / 50, the app's feature layout."""
    rms = envelope["features"]["rms"]
    zc = envelope["features"]["zero_crossings"]
    return [float(v) for v in rms] + [float(v) / ZC_NORM_SCALE for v in zc]


def rolling_mean(x: np.ndarray, m: int) -> np.ndarray:
    """Mean of the last m values; NaN until m values have been seen."""
    x = np.asarray(x, dtype=np.float64)
    out = np.full(len(x), np.nan)
    if len(x) >= m:
        c = np.concatenate([[0.0], np.cumsum(x)])
        out[m - 1:] = (c[m:] - c[:-m]) / m
    return out


def persistence(flags: np.ndarray, k: int, n: int) -> np.ndarray:
    """True where at least k of the last n flags are true."""
    f = np.asarray(flags, dtype=np.int64)
    c = np.concatenate([[0], np.cumsum(f)])
    idx = np.arange(1, len(f) + 1)
    lo = np.maximum(0, idx - n)
    return (c[idx] - c[lo]) >= k


def calibrate_threshold(streams: Sequence[np.ndarray], k: int, n: int, target: float) -> float:
    """Smallest threshold t such that persistence(stream > t, k, n) is true on
    at most *target* of the valid (non NaN) values, pooled over *streams*."""
    vals = np.concatenate([s[~np.isnan(s)] for s in streams]) if streams else np.array([])
    if len(vals) == 0:
        return float("inf")
    for t in np.unique(np.quantile(vals, np.linspace(0.5, 1.0, 501))):
        hits, total = 0, 0
        for s in streams:
            ok = ~np.isnan(s)
            a = persistence(np.where(ok, s, -np.inf) > t, k, n)
            hits += int(a[ok].sum())
            total += int(ok.sum())
        if total and hits / total <= target:
            return float(t)
    return float(vals.max() * 1.01)


def _transform(X: np.ndarray, n_ch: int) -> np.ndarray:
    return np.hstack([np.log(np.maximum(X[:, :n_ch], 0.0) + LOG_EPS), X[:, n_ch:]])


def fit_stats(X: np.ndarray, y: np.ndarray, shrinkage: float = 0.1, ridge: float = 1.0) -> dict:
    """Enrollment statistics: feature scaling, gesture means, pooled precision
    matrix, and the zero crossing to level regression."""
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    n_ch = X.shape[1] // 2
    Z = _transform(X, n_ch)
    mu = Z.mean(0)
    sd = Z.std(0) + 1e-6
    Zs = (Z - mu) / sd
    classes = np.unique(y)
    means = np.array([Zs[y == c].mean(0) for c in classes])
    resid = Zs - means[np.searchsorted(classes, y)]
    cov = np.cov(resid.T) if len(resid) > 1 else np.eye(Zs.shape[1])
    cov = (1 - shrinkage) * cov + shrinkage * np.eye(len(cov)) * np.trace(cov) / len(cov)
    prec = np.linalg.inv(cov)

    level = Z[:, :n_ch].mean(1)
    zc = Zs[:, n_ch:]
    A = np.hstack([np.ones((len(zc), 1)), zc, zc ** 2])
    reg = ridge * np.eye(A.shape[1])
    reg[0, 0] = 0.0
    coef = np.linalg.solve(A.T @ A + reg, A.T @ level)
    return {"n_ch": n_ch, "mu": mu, "sd": sd, "means": means, "prec": prec, "coef": coef}


def window_scores(stats: dict, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per window: (pattern distance, level residual in log units)."""
    X = np.atleast_2d(np.asarray(X, dtype=np.float64))
    n_ch = stats["n_ch"]
    Z = _transform(X, n_ch)
    Zs = (Z - stats["mu"]) / stats["sd"]
    D = Zs[:, None, :] - stats["means"][None, :, :]
    q = np.einsum("nkd,de,nke->nk", D, stats["prec"], D)
    pattern = np.sqrt(np.maximum(q.min(1), 0.0))
    zc = Zs[:, n_ch:]
    A = np.hstack([np.ones((len(zc), 1)), zc, zc ** 2])
    level = Z[:, :n_ch].mean(1) - A @ stats["coef"]
    return pattern, level


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------

class FeatureDriftDetector:
    """Label free drift detector on window features. See the module docstring."""

    def __init__(
        self,
        pattern_window: int = 20,
        persist_k: int = 15,
        persist_n: int = 20,
        level_window: int = 200,
        target_false_alarm: float = 0.02,
        level_margin: float = 1.25,
        min_level_threshold: float = 0.15,
        shrinkage: float = 0.1,
        enroll_windows: int = 600,
        n_groups: int = 4,
    ) -> None:
        self.pattern_window = pattern_window
        self.persist_k = persist_k
        self.persist_n = persist_n
        self.level_window = level_window
        self.target_false_alarm = target_false_alarm
        self.level_margin = level_margin
        self.min_level_threshold = min_level_threshold
        self.shrinkage = shrinkage
        self.enroll_windows = enroll_windows
        self.n_groups = n_groups

        self.stats: Optional[dict] = None
        self.pattern_threshold = float("inf")
        self.level_threshold = float("inf")
        self.calibration = "none"

        # online enrollment (see observe())
        self._enroll_X: list[list[float]] = []
        self._enroll_y: list[str] = []
        self._enroll_g: list[int] = []
        self._run_label: Optional[str] = None
        self._run_count: dict[str, int] = {}

        self.reset_stream()

    # -- fitting -------------------------------------------------------------

    @property
    def fitted(self) -> bool:
        return self.stats is not None

    def fit(self, X, y, groups=None) -> "FeatureDriftDetector":
        """Fit enrollment statistics on clean windows and calibrate both
        thresholds by cross fitting over *groups* (for example repetition
        numbers). Without at least two groups the thresholds are calibrated
        in sample, which is optimistic."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        pat_streams: list[np.ndarray] = []
        lvl_streams: list[np.ndarray] = []
        ug = np.unique(groups) if groups is not None else np.array([])
        if len(ug) >= 2:
            groups = np.asarray(groups)
            for g in ug:
                fit_m = groups != g
                if len(np.unique(y[fit_m])) < 1:
                    continue
                ho = (groups == g) & np.isin(y, np.unique(y[fit_m]))
                if not ho.any():
                    continue
                st = fit_stats(X[fit_m], y[fit_m], self.shrinkage)
                p, l = window_scores(st, X[ho])
                pat_streams.append(rolling_mean(p, self.pattern_window))
                lvl_streams.append(np.abs(rolling_mean(l, self.level_window)))
            self.calibration = "cross_fit"
        self.stats = fit_stats(X, y, self.shrinkage)
        if not pat_streams:
            p, l = window_scores(self.stats, X)
            pat_streams = [rolling_mean(p, self.pattern_window)]
            lvl_streams = [np.abs(rolling_mean(l, self.level_window))]
            self.calibration = "in_sample"
        self.pattern_threshold = calibrate_threshold(
            pat_streams, self.persist_k, self.persist_n, self.target_false_alarm / 2
        )
        lvl_vals = [s[~np.isnan(s)] for s in lvl_streams]
        lvl_vals = [v for v in lvl_vals if len(v)]
        # No held out stream as long as level_window means the level check
        # cannot be calibrated, so it stays off (threshold inf).
        self.level_threshold = (
            max(self.min_level_threshold, float(self.level_margin * max(v.max() for v in lvl_vals)))
            if lvl_vals else float("inf")
        )
        self.reset_stream()
        return self

    def observe(self, vec: Sequence[float], label: Optional[str] = None) -> bool:
        """Online use. Before the detector is fitted, labeled clean windows are
        collected as enrollment (grouped by how many times that gesture has
        come round, so the groups behave like repetitions) and the detector
        fits itself once *enroll_windows* have been collected. After that,
        every window is scored and *label* is ignored. Returns the alarm."""
        if self.fitted:
            return self.update(vec)
        if label is None:
            return False
        if label != self._run_label:
            self._run_count[label] = self._run_count.get(label, 0) + 1
            self._run_label = label
        self._enroll_X.append([float(v) for v in vec])
        self._enroll_y.append(label)
        self._enroll_g.append((self._run_count[label] - 1) % self.n_groups)
        if len(self._enroll_X) >= self.enroll_windows:
            self.fit(self._enroll_X, self._enroll_y, self._enroll_g)
            self._enroll_X, self._enroll_y, self._enroll_g = [], [], []
        return False

    # -- streaming -------------------------------------------------------------

    def reset_stream(self) -> None:
        self._pat_buf: deque[float] = deque(maxlen=self.pattern_window)
        self._lvl_buf: deque[float] = deque(maxlen=self.level_window)
        self._flag_buf: deque[bool] = deque(maxlen=self.persist_n)
        self.pattern_score: Optional[float] = None
        self.level_score: Optional[float] = None
        self.pattern_alarm = False
        self.level_alarm = False

    @property
    def alarm(self) -> bool:
        return self.pattern_alarm or self.level_alarm

    @property
    def reason(self) -> Optional[str]:
        """'pattern' (channel pattern changed, e.g. electrodes moved),
        'level' (overall amplitude changed, e.g. contact), or None."""
        if self.pattern_alarm:
            return "pattern"
        if self.level_alarm:
            return "level"
        return None

    def update(self, vec: Sequence[float]) -> bool:
        """Score one window and return whether the alarm is on."""
        if not self.fitted:
            raise RuntimeError("FeatureDriftDetector.update() before fit()")
        p, l = window_scores(self.stats, np.asarray(vec, dtype=np.float64)[None, :])
        self._pat_buf.append(float(p[0]))
        self._lvl_buf.append(float(l[0]))
        if len(self._pat_buf) == self.pattern_window:
            self.pattern_score = sum(self._pat_buf) / self.pattern_window
            self._flag_buf.append(self.pattern_score > self.pattern_threshold)
        else:
            self._flag_buf.append(False)
        self.pattern_alarm = sum(self._flag_buf) >= self.persist_k
        if len(self._lvl_buf) == self.level_window:
            self.level_score = sum(self._lvl_buf) / self.level_window
            self.level_alarm = abs(self.level_score) > self.level_threshold
        else:
            self.level_alarm = False
        return self.alarm

    def run(self, X) -> dict:
        """Vectorised equivalent of calling update() on every row of *X*
        from a fresh stream. Returns per window arrays."""
        if not self.fitted:
            raise RuntimeError("FeatureDriftDetector.run() before fit()")
        p, l = window_scores(self.stats, X)
        pm = rolling_mean(p, self.pattern_window)
        lm = rolling_mean(l, self.level_window)
        pat_alarm = persistence(np.where(np.isnan(pm), -np.inf, pm) > self.pattern_threshold,
                                self.persist_k, self.persist_n)
        lvl_alarm = np.where(np.isnan(lm), 0.0, np.abs(lm)) > self.level_threshold
        return {
            "pattern_score": pm,
            "level_score": lm,
            "pattern_alarm": pat_alarm,
            "level_alarm": lvl_alarm,
            "alarm": pat_alarm | lvl_alarm,
        }

    def summary(self) -> dict:
        return {
            "fitted": self.fitted,
            "calibration": self.calibration,
            "pattern_threshold": self.pattern_threshold,
            "level_threshold": self.level_threshold,
        }
