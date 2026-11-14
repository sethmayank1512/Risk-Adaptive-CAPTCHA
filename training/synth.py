"""Synthetic trajectories.

Two uses:
 1. Bot *tiers* (naive -> noisy Bezier -> replay) to augment bot diversity and to
    run leave-one-bot-type-out evaluation. These are generated from scratch and
    are meant to be used alongside real data.
 2. `write_fake_*` helpers that emit tiny stand-in files in the formats the
    loaders expect, so the whole pipeline can be smoke-tested without the real
    datasets. Models trained from these are tagged "synthetic" and must never
    be reported as results.

All trajectories are (x, y, t_ms) with x, y normalised to [0,1].
"""
import base64
import json
import struct
from pathlib import Path

import numpy as np

W, H = 1920, 1080


def _quant(x, y):
    return np.round(x * W) / W, np.round(y * H) / H


def _minjerk(s):
    return 10 * s ** 3 - 15 * s ** 4 + 6 * s ** 5


def synth_human(rng, n_strokes=None):
    """Human-like pointer path: minimum-jerk strokes, curvature, corrections, pauses, irregular events."""
    pos = rng.uniform(0.1, 0.9, 2)
    segs = []
    for _ in range(int(n_strokes or rng.integers(1, 5))):
        tgt = np.clip(pos + rng.normal(0, 0.25, 2), 0.03, 0.97)
        vec = tgt - pos
        dist = float(np.hypot(*vec)) + 1e-6
        dur = (0.25 + 0.9 * dist ** 0.6) * rng.uniform(0.7, 1.4)
        n = max(int(dur * 1000), 50)
        s = np.linspace(0, 1, n)
        base = pos + np.outer(_minjerk(s), vec)
        perp = np.array([-vec[1], vec[0]]) / dist
        base += np.outer(rng.normal(0, 0.04) * dist * np.sin(np.pi * s), perp)
        if rng.random() < 0.3:  # overshoot then correct
            over = tgt + vec / dist * rng.uniform(0.005, 0.03)
            c = np.linspace(0, 1, int(rng.uniform(80, 200)))
            base = np.vstack([base, base[-1] + np.outer(_minjerk(c), over - base[-1]),
                              over + np.outer(_minjerk(c), tgt - over)])
        pos = base[-1]
        segs.append(base)
        if rng.random() < 0.7:
            segs.append(np.tile(pos, (int(rng.uniform(60, 600)), 1)))
    dense = np.vstack(segs)
    dense[:, 0] += rng.normal(0, 0.0003, len(dense)).cumsum() * 0.02  # slow drift
    ts, t = [], 0.0
    while t < len(dense) - 1:
        ts.append(t)
        t += rng.uniform(7, 17)
    ts = np.array(ts)
    x = np.interp(ts, np.arange(len(dense)), dense[:, 0])
    y = np.interp(ts, np.arange(len(dense)), dense[:, 1])
    x, y = _quant(x, y)
    return np.column_stack([x, y, ts + rng.uniform(100, 800)])


def _bezier(p0, p1, p2, p3, u):
    u = u[:, None]
    return ((1 - u) ** 3) * p0 + 3 * ((1 - u) ** 2) * u * p1 + 3 * (1 - u) * u ** 2 * p2 + u ** 3 * p3


def synth_bot(rng, kind):
    p0, p3 = rng.uniform(0.05, 0.95, 2), rng.uniform(0.05, 0.95, 2)
    dur = rng.uniform(300, 1500)
    if kind == "naive":            # few events, straight line (e.g. WebDriver moveTo)
        n = int(rng.integers(2, 12))
        u = np.linspace(0, 1, n)
        t = np.linspace(0, dur, n)
        xy = p0 + np.outer(u, p3 - p0)
        x, y = _quant(xy[:, 0], xy[:, 1])
    elif kind == "naive_dense":    # straight line, constant velocity, exact 60 Hz (e.g. pyautogui)
        n = max(int(dur / 16.67), 10)
        u = np.linspace(0, 1, n)
        t = np.arange(n) * 16.67
        xy = p0 + np.outer(u, p3 - p0)
        x, y = _quant(xy[:, 0], xy[:, 1])
    elif kind == "bezier":         # humanising library: Bezier, constant parameter speed
        n = max(int(dur / 16.67), 10)
        c1, c2 = p0 + rng.normal(0, 0.15, 2), p3 + rng.normal(0, 0.15, 2)
        xy = _bezier(p0, c1, c2, p3, np.linspace(0, 1, n))
        t = np.arange(n) * 16.67
        x, y = _quant(xy[:, 0], xy[:, 1])
    elif kind == "noisy_bezier":   # Bezier + ease-in/out + jitter + irregular timing + a pause
        n = max(int(dur / 12), 15)
        c1, c2 = p0 + rng.normal(0, 0.15, 2), p3 + rng.normal(0, 0.15, 2)
        xy = _bezier(p0, c1, c2, p3, _minjerk(np.linspace(0, 1, n)))
        xy += rng.normal(0, 0.0007, xy.shape)
        t = np.cumsum(rng.uniform(7, 17, n))
        k = int(rng.integers(n // 3, 2 * n // 3))
        t[k:] += rng.uniform(80, 300)
        x, y = _quant(xy[:, 0], xy[:, 1])
    else:
        raise ValueError(kind)
    return np.column_stack([x, y, t + rng.uniform(100, 800)])


def synth_replay(rng, human):
    """Replay of a real/synthetic human trajectory with small shift, time-warp and noise."""
    h = np.array(human, dtype=float).copy()
    h[:, 0] = np.clip(h[:, 0] + rng.uniform(-0.01, 0.01), 0, 1)
    h[:, 1] = np.clip(h[:, 1] + rng.uniform(-0.01, 0.01), 0, 1)
    h[:, 2] = (h[:, 2] - h[0, 2]) * rng.uniform(0.9, 1.1) + h[0, 2]
    h[:, 0] += rng.normal(0, 0.5 / W, len(h))
    h[:, 1] += rng.normal(0, 0.5 / H, len(h))
    return h


BOT_KINDS = ["naive", "naive_dense", "bezier", "noisy_bezier"]


# ----------------------------------------------------------------- fake files
def write_fake_delbot(path, n=300, seed=0):
    """Tiny DELBOT-like JSON: {'humans': [...], 'bots': [...]} - ONLY for smoke tests."""
    rng = np.random.default_rng(seed)
    def pack(traj, i, extra=None):
        d = {"id": i, "mouse": [{"x": float(a), "y": float(b), "t": float(c)} for a, b, c in traj]}
        d.update(extra or {})
        return d
    humans = [pack(synth_human(rng), f"h{i}") for i in range(n)]
    bots = [pack(synth_bot(rng, BOT_KINDS[i % len(BOT_KINDS)]), f"b{i}", {"type": BOT_KINDS[i % len(BOT_KINDS)]})
            for i in range(n)]
    Path(path).write_text(json.dumps({"humans": humans, "bots": bots}))


def write_fake_capycap_stream(path, n=200, seed=1, canvas=(800, 600), hz=240, stride12=True):
    """Tiny CaptchaSolve30k-like JSONL (base64 of <ffB[3x]> records, pixel coords) - ONLY for smoke tests.

    Humans here are deliberately a bit different from the DELBOT-like ones
    (fixed 240 Hz ticks, longer sessions, different canvas) to mimic domain shift.
    """
    rng = np.random.default_rng(seed)
    fmt = "<ffB3x" if stride12 else "<ffB"
    with open(path, "w") as fh:
        for i in range(n):
            tr = synth_human(rng, n_strokes=int(rng.integers(3, 9)))
            tt = np.arange(tr[0, 2], tr[-1, 2], 1000.0 / hz)
            x = np.interp(tt, tr[:, 2], tr[:, 0]) * canvas[0]
            y = np.interp(tt, tr[:, 2], tr[:, 1]) * canvas[1]
            down = (rng.random(len(tt)) < 0.4).astype(int)
            blob = b"".join(struct.pack(fmt, float(a), float(b), int(c)) for a, b, c in zip(x, y, down))
            rec = {"id": f"c{i}", "gameType": "thread-the-needle", "duration": float(tt[-1] - tt[0]),
                   "pointerData": base64.b64encode(blob).decode()}
            fh.write(json.dumps(rec) + "\n")


def write_fake_capycap(path, n=200, seed=1, hz=240, touch_frac=0.1):
    """Fake CaptchaSolve30k JSONL in the DOCUMENTED format: tickInputs on a 200x200 grid at 240 Hz,
    leading (0,0) placeholder ticks, some touchscreen sessions. ONLY for smoke tests."""
    rng = np.random.default_rng(seed)
    with open(path, "w") as fh:
        for i in range(n):
            tr = synth_human(rng, n_strokes=int(rng.integers(3, 9)))
            tt = np.arange(tr[0, 2], tr[-1, 2], 1000.0 / hz)
            x = np.interp(tt, tr[:, 2], tr[:, 0]) * 200
            y = np.interp(tt, tr[:, 2], tr[:, 1]) * 200
            lead = int(rng.integers(20, 100))
            ticks = [{"isDown": False, "sampleIndex": 1, "x": 0.0, "y": 0.0} for _ in range(lead)]
            ticks += [{"isDown": bool(rng.random() < 0.4), "sampleIndex": k + 1, "x": float(a), "y": float(b)}
                      for k, (a, b) in enumerate(zip(x, y))]
            rec = {"index": i, "tickInputs": ticks, "inputStream": "AAAA", "inputSampleCount": len(ticks) // 2,
                   "duration": int(len(ticks) * 1000 / hz), "touchscreen": bool(rng.random() < touch_frac),
                   "gameType": "thread-the-needle", "physicsTickCount": len(ticks), "puzzleSeed": int(rng.integers(0, 3000))}
            fh.write(json.dumps(rec) + "\n")


def write_fake_delbot_tree(root, n=300, seed=0):
    """Fake DELBOT layout: python/sessions.json = {'human': [rel paths], 'bot': [rel paths]} plus text files
    ('t x y Move' lines, pixel coords) under python/dataset/. Returns the sessions.json path. ONLY for smoke tests."""
    rng = np.random.default_rng(seed)
    base = Path(root) / "python"
    idx = {"human": [], "bot": []}
    def dump(traj, rel):
        fp = base / rel
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_text("\n".join(f"{t:.1f} {int(round(x * W))} {int(round(y * H))} Move" for x, y, t in traj))
    for i in range(n):
        rel = f"dataset/human/h{i}.txt"; dump(synth_human(rng), rel); idx["human"].append(rel)
    for i in range(n):
        k = BOT_KINDS[i % len(BOT_KINDS)]
        rel = f"dataset/bots/{k}/b{i}.txt"; dump(synth_bot(rng, k), rel); idx["bot"].append(rel)
    (base / "sessions.json").write_text(json.dumps(idx))
    return str(base / "sessions.json")
