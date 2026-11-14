"""Trajectory feature extraction.

The SAME function is used for training (public datasets) and serving (live
Flask sessions), so train/serve skew cannot come from feature code.

Input : array-like of (x, y, t_ms) with x, y normalised to [0, 1] by screen
        (or canvas) width/height.
Output: dict of features, or None when there is too little evidence.

Two feature groups:
  INVARIANT_FEATURES - computed after resampling to a fixed 50 Hz timeline, so
      they do not depend on the recording device's sampling rate. These are
      the ones used for cross-dataset training.
  TIMING_FEATURES    - computed from the raw event timestamps. They differ
      wildly between datasets (240 Hz ticks vs browser mouse events), so they
      are NOT used by the default model; they are kept for analysis only.
"""
import numpy as np

RESAMPLE_HZ = 50.0
MIN_POINTS = 8
MIN_DURATION_MS = 150.0
MIN_PATH = 1e-3          # total path (in screen widths) below which there is no movement evidence
STATIONARY_STEP = 2e-4   # per-sample movement below this counts as "not moving"

INVARIANT_FEATURES = [
    "duration_s", "path_len", "straightness", "line_dev_rms", "win_straightness",
    "v_mean", "v_std", "v_cv", "v_median", "v_p90", "v_max_log",
    "acc_std", "acc_abs_mean", "acc_max_log",
    "jerk_abs_mean_log", "jerk_std_log",
    "turn_abs_mean", "turn_std", "turn_sharp_frac", "turn_zero_frac",
    "stationary_frac", "pauses_per_s", "speed_peaks_per_s",
    "micro_jitter", "x_range", "y_range",
]
TIMING_FEATURES = ["raw_n_events", "raw_dt_mean_ms", "raw_dt_cv", "raw_dt_min_ms"]
ALL_FEATURES = INVARIANT_FEATURES + TIMING_FEATURES


def clean(traj):
    """Return a float array (N,3) sorted by time with unique timestamps, or None."""
    try:
        a = np.asarray(traj, dtype=float)
    except (ValueError, TypeError):
        return None
    if a.ndim != 2 or a.shape[1] < 3 or len(a) == 0:
        return None
    a = a[:, :3]
    a = a[np.isfinite(a).all(axis=1)]
    if len(a) == 0:
        return None
    a = a[np.argsort(a[:, 2], kind="stable")]
    # keep the last sample for duplicated timestamps
    keep = np.append(np.diff(a[:, 2]) > 0, True)
    return a[keep]


def resample(a, hz=RESAMPLE_HZ):
    t0, t1 = a[0, 2], a[-1, 2]
    n = int((t1 - t0) / 1000.0 * hz) + 1
    tt = t0 + np.arange(n) * (1000.0 / hz)
    return np.column_stack([np.interp(tt, a[:, 2], a[:, 0]), np.interp(tt, a[:, 2], a[:, 1])])


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def _runs(mask, min_len):
    """Number of runs of True with length >= min_len."""
    n = run = 0
    for m in mask:
        if m:
            run += 1
        else:
            n += run >= min_len
            run = 0
    return n + (run >= min_len)


def extract(traj):
    a = clean(traj)
    if a is None or len(a) < MIN_POINTS:
        return None
    dur_ms = a[-1, 2] - a[0, 2]
    if dur_ms < MIN_DURATION_MS:
        return None
    raw_dt = np.diff(a[:, 2])
    xy = resample(a)
    if len(xy) < 8:
        return None
    dt = 1.0 / RESAMPLE_HZ
    d = np.diff(xy, axis=0)
    step = np.hypot(d[:, 0], d[:, 1])
    path = float(step.sum())
    if path < MIN_PATH:
        return None
    disp = float(np.hypot(*(xy[-1] - xy[0])))
    v = step / dt
    acc = np.diff(v) / dt
    jerk = np.diff(acc) / dt
    eps = 1e-9

    f = {}
    f["duration_s"] = dur_ms / 1000.0
    f["path_len"] = path
    f["straightness"] = disp / path
    # deviation from the start->end chord, relative to chord length
    if disp > 1e-3:
        u = (xy[-1] - xy[0]) / disp
        rel = xy - xy[0]
        cross = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
        f["line_dev_rms"] = float(min(np.sqrt(np.mean(cross ** 2)) / disp, 10.0))
    else:
        f["line_dev_rms"] = 10.0  # returned to start: very curved
    # straightness inside 0.5 s windows (humans curve locally, naive bots do not)
    w = max(int(0.5 * RESAMPLE_HZ), 3)
    ws = []
    for i in range(0, len(step) - w + 1, w):
        p = step[i:i + w].sum()
        if p > 1e-3:
            ws.append(np.hypot(*(xy[i + w] - xy[i])) / p)
    f["win_straightness"] = float(np.mean(ws)) if ws else 1.0

    f["v_mean"] = float(v.mean())
    f["v_std"] = float(v.std())
    f["v_cv"] = float(v.std() / (v.mean() + eps))
    f["v_median"] = float(np.median(v))
    f["v_p90"] = float(np.percentile(v, 90))
    f["v_max_log"] = float(np.log1p(v.max()))
    f["acc_std"] = float(acc.std()) if len(acc) else 0.0
    f["acc_abs_mean"] = float(np.abs(acc).mean()) if len(acc) else 0.0
    f["acc_max_log"] = float(np.log1p(np.abs(acc).max())) if len(acc) else 0.0
    f["jerk_abs_mean_log"] = float(np.log1p(np.abs(jerk).mean())) if len(jerk) else 0.0
    f["jerk_std_log"] = float(np.log1p(jerk.std())) if len(jerk) else 0.0

    mv = step > STATIONARY_STEP
    theta = np.arctan2(d[mv, 1], d[mv, 0])
    turn = np.abs(_wrap(np.diff(theta))) if len(theta) > 2 else np.zeros(1)
    f["turn_abs_mean"] = float(turn.mean())
    f["turn_std"] = float(turn.std())
    f["turn_sharp_frac"] = float((turn > np.pi / 4).mean())
    f["turn_zero_frac"] = float((turn < 1e-3).mean())  # perfectly straight consecutive steps

    still = ~mv
    f["stationary_frac"] = float(still.mean())
    f["pauses_per_s"] = _runs(still, int(0.1 * RESAMPLE_HZ)) / (dur_ms / 1000.0)
    sm = np.convolve(v, np.ones(3) / 3, mode="same")
    pk = (sm[1:-1] > sm[:-2]) & (sm[1:-1] >= sm[2:]) & (sm[1:-1] > 0.5 * sm.max())
    f["speed_peaks_per_s"] = float(pk.sum() / (dur_ms / 1000.0))
    dd = np.diff(xy, 2, axis=0)
    f["micro_jitter"] = float(np.hypot(dd[:, 0], dd[:, 1]).mean() / (step.mean() + eps)) if len(dd) else 0.0
    f["x_range"] = float(np.ptp(xy[:, 0]))
    f["y_range"] = float(np.ptp(xy[:, 1]))

    f["raw_n_events"] = float(len(a))
    f["raw_dt_mean_ms"] = float(raw_dt.mean())
    f["raw_dt_cv"] = float(raw_dt.std() / (raw_dt.mean() + eps))
    f["raw_dt_min_ms"] = float(raw_dt.min())
    return f


def last_window(traj, window_s=6.0):
    """Serving-time helper: keep the final `window_s` seconds of a trajectory."""
    a = clean(traj)
    if a is None:
        return None
    return a[a[:, 2] >= a[-1, 2] - window_s * 1000.0]
