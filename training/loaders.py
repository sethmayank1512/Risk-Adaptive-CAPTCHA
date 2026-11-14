"""Dataset loaders -> list[Session].

IMPORTANT: the exact on-disk schemas of DELBOT-Mouse (python/sessions.json) and
CaptchaSolve30k (binary pointer blob inside JSONL) were NOT verified against the
real files when this code was written. Both loaders therefore auto-detect the
structure and raise a descriptive error if they cannot. Run
    python -m training.inspect_data --delbot PATH --capycap PATH
first, and if detection is wrong, pass the override flags (see train.py --help)
or adapt `_as_points` / `decode_blob`, which are the only schema-specific parts.
"""
import base64
import binascii
import json
import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class Session:
    traj: np.ndarray          # (N,3) x,y in [0,1], t in ms
    label: int                # 0 human, 1 bot
    source: str               # 'delbot' | 'capycap' | 'synthetic' | 'collected'
    bot_type: str = "human"
    group: str = ""           # split unit - windows from one session never straddle splits
    meta: dict = field(default_factory=dict)


# ----------------------------------------------------------------- DELBOT
T_KEYS = ("t", "time", "timestamp", "ts", "timems", "time_ms", "date")
X_KEYS, Y_KEYS = ("x", "clientx", "pagex", "screenx"), ("y", "clienty", "pagey", "screeny")
POINT_KEYS = ("mouse", "movements", "moves", "mousemoves", "events", "points", "trajectory",
              "positions", "coords", "record", "data", "mouse_events")
GROUP_KEYS = ("user", "userid", "user_id", "session", "sessionid", "session_id", "id", "uid")


def _lk(d):
    return {str(k).lower(): k for k in d}


def _pt_from_dict(d):
    lk = _lk(d)
    def g(keys):
        for k in keys:
            if k in lk and isinstance(d[lk[k]], (int, float)):
                return float(d[lk[k]])
        return None
    x, y, t = g(X_KEYS), g(Y_KEYS), g(T_KEYS)
    if x is None or y is None or t is None:
        return None
    return (x, y, t)


def _as_points(obj, assume_dt_ms=None):
    """Return (N,3) array if obj looks like a list of mouse samples, else None."""
    if not isinstance(obj, list) or len(obj) < 2:
        return None
    first = obj[0]
    rows = []
    if isinstance(first, dict):
        for p in obj:
            if not isinstance(p, dict):
                return None
            typ = str(p.get(next((k for k in p if str(k).lower() == "type"), ""), "move")).lower()
            if "move" not in typ and typ != "":
                continue                      # ignore clicks/keys mixed into the same stream
            r = _pt_from_dict(p)
            if r is None:
                return None
            rows.append(r)
    elif isinstance(first, (list, tuple)) and len(first) >= 3 and all(isinstance(v, (int, float)) for v in first[:3]):
        rows = [tuple(map(float, p[:3])) for p in obj if isinstance(p, (list, tuple)) and len(p) >= 3]
    elif isinstance(first, (list, tuple)) and len(first) == 2 and assume_dt_ms:
        rows = [(float(p[0]), float(p[1]), i * assume_dt_ms) for i, p in enumerate(obj)]
    else:
        return None
    return np.array(rows, dtype=float) if len(rows) >= 2 else None


def _label_from_key(k, v=None):
    """Infer label (1=bot, 0=human) from a dict key (and optional scalar value)."""
    k = str(k).lower()
    if v is None:
        if "bot" in k and "human" not in k:
            return 1
        if any(s in k for s in ("human", "real", "genuine")):
            return 0
        return None
    if isinstance(v, bool):
        if k in ("isbot", "is_bot", "bot"):
            return int(v)
        if k in ("ishuman", "is_human", "human"):
            return int(not v)
    if isinstance(v, str):
        s = v.lower()
        if "bot" in s or "fake" in s or "gan" in s or "synthetic" in s:
            return 1
        if any(x in s for x in ("human", "real", "genuine")):
            return 0
    if isinstance(v, (int, float)) and k in ("label", "class", "y", "target", "isbot", "is_bot"):
        return int(v == 1)  # assumption: 1 == bot. Verify with inspect_data.py!
    return None


def _walk(obj, label, bot_type, out, gid, assume_dt_ms, depth=0):
    if depth > 8:
        return
    pts = _as_points(obj, assume_dt_ms)
    if pts is not None:
        if label is not None:
            out.append((pts, label, bot_type, gid))
        return
    if isinstance(obj, dict):
        lab, bt = label, bot_type
        for k, v in obj.items():
            if not isinstance(v, (list, dict)):
                l2 = _label_from_key(k, v)
                if l2 is not None:
                    lab = l2
                    if isinstance(v, str) and l2 == 1:
                        bt = v
        if lab == 1:  # a session-level 'type'/'kind' string names the bot family
            for tk in ("bot_type", "type", "kind", "category", "generator", "method"):
                tv = next((v for k, v in obj.items() if str(k).lower() == tk and isinstance(v, str)), None)
                if tv and "human" not in tv.lower():
                    bt = tv
                    break
        g = gid
        lk = _lk(obj)
        for gk in GROUP_KEYS:
            if gk in lk and isinstance(obj[lk[gk]], (str, int)):
                g = str(obj[lk[gk]])
                break
        for pk in POINT_KEYS:
            if pk in lk:
                pts = _as_points(obj[lk[pk]], assume_dt_ms)
                if pts is not None and lab is not None:
                    out.append((pts, lab, bt, g))
                    return
        for k, v in obj.items():
            if isinstance(v, (list, dict)):
                lk_label = _label_from_key(k)
                _walk(v, lk_label if lk_label is not None else lab,
                      str(k) if lk_label == 1 else bt, out, g, assume_dt_ms, depth + 1)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _walk(v, label, bot_type, out, f"{gid}/{i}" if gid else str(i), assume_dt_ms, depth + 1)


GENERIC_DIRS = {"bot", "bots", "human", "humans", "data", "dataset", "datasets", "python", "trajectories", "."}


def _kv_point(line):
    pairs = dict((k.lower(), v) for k, v in re.findall(r"([A-Za-z_]+)\s*[:=]\s*(-?[\d.]+(?:[eE][-+]?\d+)?)", line))
    if len(pairs) < 3:
        return None
    return _pt_from_dict({k: float(v) for k, v in pairs.items()})


def parse_text_trajectory(text, layout="auto"):
    """Parse one DELBOT-style trajectory text file -> ((N,3) array [x,y,t], layout) or (None, None).

    Handles: JSON arrays, 'key:value' lines, and whitespace/comma separated numbers with an optional
    event-type word (non-'move' events such as clicks are dropped).
    Column layout is inferred: (t,x,y) if the first column is non-decreasing, else (x,y,t).
    """
    s = text.strip()
    if not s:
        return None, None
    if s[0] in "[{":
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                obj = next((v for v in obj.values() if isinstance(v, list)), None)
            pts = _as_points(obj)
            if pts is not None:
                return pts, "json"
        except (ValueError, TypeError):
            pass
    rows = []
    for line in s.splitlines():
        line = line.strip()
        if not line:
            continue
        kv = _kv_point(line)
        if kv is not None:
            rows.append(("kv",) + kv)
            continue
        nums, typ = [], None
        for tok in re.split(r"[,\s;|]+", line):
            if not tok:
                continue
            try:
                nums.append(float(tok))
            except ValueError:
                typ = tok.lower()
        if len(nums) >= 3 and (typ is None or "move" in typ):
            rows.append(("num",) + tuple(nums[:3]))
    if len(rows) < 2:
        return None, None
    if rows[0][0] == "kv":
        return np.array([r[1:] for r in rows if r[0] == "kv"], float), "key:value"
    R = np.array([r[1:] for r in rows if r[0] == "num"], float)
    if len(R) < 2:
        return None, None
    if layout == "txy":
        return R[:, [1, 2, 0]], "t,x,y (forced)"
    if layout == "xyt":
        return R[:, [0, 1, 2]], "x,y,t (forced)"
    mono = lambda c: (np.diff(R[:, c]) >= 0).mean() >= 0.99
    if mono(0):
        return R[:, [1, 2, 0]], "t,x,y"
    if mono(2):
        return R[:, [0, 1, 2]], "x,y,t"
    return None, None


def _bot_type_from_path(rel):
    parts = [x for x in Path(rel).parts[:-1] if x.lower() not in GENERIC_DIRS]
    return parts[-1] if parts else "bot"


def _load_delbot_paths(jpath, d, max_sessions, layout="auto"):
    """sessions.json = {'human': [relative paths], 'bot': [relative paths]} -> raw list."""
    base = Path(jpath).parent
    raw, missing, layouts, bad = [], 0, {}, 0
    for key, items in d.items():
        lab = _label_from_key(key)
        if lab is None or not isinstance(items, list):
            continue
        for i, rel in enumerate(items):
            if max_sessions and len(raw) >= max_sessions:
                break
            if not isinstance(rel, str):
                continue
            parts = [x for x in Path(rel).parts if x not in ("..", ".")]
            if parts and parts[0] == "python":
                parts = parts[1:]
            cands = [base / Path(*parts)] if parts else []
            cands += [base / rel, base.parent / rel, base / Path(rel).name]
            fp = next((q for q in cands if q.is_file()), None)
            if fp is not None:
                text = fp.read_text(errors="ignore")
            elif len(rel) > 200 or "\n" in rel:
                text = rel                                  # trajectory stored inline in the JSON
            else:
                missing += 1
                continue
            pts, lay = parse_text_trajectory(text, layout)
            if pts is None or len(pts) < 2:
                bad += 1
                continue
            layouts[lay] = layouts.get(lay, 0) + 1
            raw.append((pts, lab, "human" if lab == 0 else _bot_type_from_path(rel), rel))
    total = sum(len(v) for v in d.values() if isinstance(v, list))
    if not raw:
        hint = (f"{missing}/{total} paths in {jpath} do not exist. Extract the data first:\n"
                f"    cd {base} && tar -xzf dataset.tar.gz") if missing else \
               "files exist but could not be parsed - run training.inspect_data and send me the first lines"
        raise ValueError("DELBOT: no trajectories loaded. " + hint)
    print(f"[delbot] loaded {len(raw)} files ({missing} missing, {bad} unparseable); column layouts: {layouts}")
    return raw


def load_delbot(path, screen=(1920, 1080), assume_dt_ms=None, max_sessions=None, layout="auto"):
    p = Path(path)
    files = [p] if p.is_file() else sorted(list(p.rglob("sessions.json")) + list(p.rglob("*.jsonl")) or list(p.rglob("*.json")))
    if not files:
        raise FileNotFoundError(f"No .json/.jsonl files under {path}")
    raw = []
    for fp in files:
        if fp.suffix == ".jsonl":
            for i, line in enumerate(open(fp)):
                if line.strip():
                    _walk(json.loads(line), None, "bot", raw, f"{fp.stem}:{i}", assume_dt_ms)
            continue
        d = json.load(open(fp))
        if isinstance(d, dict) and any(isinstance(v, list) and v and isinstance(v[0], str) for v in d.values()):
            raw += [(pts, lab, bt, rel) for pts, lab, bt, rel in _load_delbot_paths(fp, d, max_sessions, layout)]
        else:
            _walk(d, None, "bot", raw, fp.stem, assume_dt_ms)
    if not raw:
        raise ValueError(
            "DELBOT loader found no labelled mouse trajectories. Run "
            "`python -m training.inspect_data --delbot <path>` to see the structure.")
    # pixel coordinates -> [0,1]: use one global screen estimate (at least `screen`)
    pix = [r[0] for r in raw if max(r[0][:, 0].max(), r[0][:, 1].max()) > 1.5]
    if pix:
        allx = np.concatenate([q[:, 0] for q in pix]); ally = np.concatenate([q[:, 1] for q in pix])
        W, H = max(screen[0], np.percentile(allx, 99.9)), max(screen[1], np.percentile(ally, 99.9))
        print(f"[delbot] {len(pix)} sessions in pixels; normalising by {W:.0f}x{H:.0f}")
    sessions = []
    for i, (pts, lab, bt, g) in enumerate(raw):
        if max_sessions and i >= max_sessions:
            break
        pts = pts.copy()
        if pts[:, 2].max() - pts[:, 2].min() < 5 and len(pts) > 50:   # looks like seconds
            pts[:, 2] *= 1000.0
        if max(pts[:, 0].max(), pts[:, 1].max()) > 1.5:
            pts[:, 0] /= W
            pts[:, 1] /= H
        sessions.append(Session(pts, int(lab), "delbot", "human" if lab == 0 else str(bt), f"delbot:{g}:{i}"))
    return sessions


# ----------------------------------------------------------------- CaptchaSolve30k
BLOB_KEYS = ("pointerdata", "pointer_data", "pointer", "trajectory", "samples", "replay",
             "mouse", "events", "data", "movements", "path")


def _to_bytes(v):
    if isinstance(v, (bytes, bytearray)):
        return bytes(v)
    if isinstance(v, list) and v and all(isinstance(i, int) and 0 <= i < 256 for i in v[:50]):
        return bytes(v)
    if isinstance(v, str) and len(v) > 40:
        for dec in (base64.b64decode, base64.urlsafe_b64decode):
            try:
                return dec(v + "=" * (-len(v) % 4))
            except (binascii.Error, ValueError):
                continue
    return None


_CANDIDATES = [("<ffB", 9), ("<ffB", 12), (">ffB", 9), (">ffB", 12),
               ("<iiB", 9), ("<iiB", 12), (">iiB", 9), (">iiB", 12)]


def decode_blob(blob, fmt=None, stride=None):
    """Decode packed (x, y, isDown) records. Auto-detects float32/int32, endianness and stride."""
    cands = [(fmt, stride)] if fmt and stride else _CANDIDATES
    best, best_score = None, None
    for header in (0, 4, 8):
        for f, s in cands:
            body = blob[header:]
            n = len(body) // s
            if n < 10:
                continue
            xs = np.empty(n); ys = np.empty(n); dn = np.empty(n)
            try:
                for i in range(min(n, 400)):    # validate on a prefix
                    xs[i], ys[i], dn[i] = struct.unpack_from(f, body, i * s)
            except struct.error:
                continue
            m = min(n, 400)
            xs, ys, dn = xs[:m], ys[:m], dn[:m]
            if not (np.isfinite(xs).all() and np.isfinite(ys).all()):
                continue
            if np.abs(xs).max() > 1e5 or np.abs(ys).max() > 1e5 or not np.isin(dn, (0, 1)).all():
                continue
            smooth = np.median(np.hypot(np.diff(xs), np.diff(ys))) / (np.ptp(xs) + np.ptp(ys) + 1e-9)
            if best_score is None or smooth < best_score:
                best, best_score = (header, f, s), smooth
    if best is None:
        raise ValueError("Could not decode pointer blob; pass --capycap-fmt/--capycap-stride "
                         "(see inspect_data.py output).")
    header, f, s = best
    body = blob[header:]
    n = len(body) // s
    arr = np.array([struct.unpack_from(f, body, i * s) for i in range(n)], dtype=float)
    return arr[:, 0], arr[:, 1], arr[:, 2], best


def _ticks_to_traj(rec, grid=200.0):
    """Documented format: tickInputs = [{x, y, isDown, sampleIndex}] at 240 Hz, x,y on a 0-200 grid."""
    ticks = rec.get("tickInputs")
    if not isinstance(ticks, list) or len(ticks) < 20 or not isinstance(ticks[0], dict):
        return None
    x = np.fromiter((t.get("x", 0.0) for t in ticks), float, len(ticks))
    y = np.fromiter((t.get("y", 0.0) for t in ticks), float, len(ticks))
    live = np.flatnonzero((x != 0) | (y != 0))        # leading (0,0) = pointer not yet in the game area
    if len(live) == 0:
        return None
    first = live[0]
    n_tick = rec.get("physicsTickCount") or len(ticks)
    step_ms = float(rec["duration"]) / n_tick if isinstance(rec.get("duration"), (int, float)) and rec["duration"] > 0 else 1000.0 / 240.0
    t = np.arange(len(ticks)) * step_ms
    return np.column_stack([np.clip(x[first:] / grid, 0, 1.5), np.clip(y[first:] / grid, 0, 1.5), t[first:] - t[first]])


def load_capycap(path, max_sessions=3000, canvas=None, hz=240.0, field=None, fmt=None, stride=None,
                 min_samples=240, include_touch=False, source="ticks"):
    """CaptchaSolve30k -> human Sessions (mouse only unless include_touch).

    source='ticks'  : use `tickInputs` (documented JSON, 240 Hz, 200x200 grid)  [default]
    source='stream' : decode the binary `inputStream` (format differs between file versions; kept as fallback)
    """
    p = Path(path)
    files = [p] if p.is_file() else sorted(p.glob("*.jsonl"))
    if not files:
        raise FileNotFoundError(f"No .jsonl under {path}")
    grid = float(canvas[0]) if canvas else 200.0
    sessions, skipped_touch, skipped_short = [], 0, 0
    games = {}
    for fp in files:
        with open(fp) as fh:
            for i, line in enumerate(fh):
                if max_sessions and len(sessions) >= max_sessions:
                    break
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("touchscreen") and not include_touch:
                    skipped_touch += 1
                    continue
                tr = None
                if source == "ticks":
                    tr = _ticks_to_traj(rec, grid)
                else:
                    lk = _lk(rec)
                    keys = [field] if field else [lk[k] for k in BLOB_KEYS if k in lk]
                    blob = next((b for b in (_to_bytes(rec[k]) for k in keys if k in rec) if b), None)
                    if blob is not None:
                        x, y, dn, _ = decode_blob(blob, fmt, stride)
                        tr = np.column_stack([x / grid, y / grid, np.arange(len(x)) * 1000.0 / hz])
                if tr is None or len(tr) < min_samples:
                    skipped_short += 1
                    continue
                gt = str(rec.get("gameType", ""))
                games[gt] = games.get(gt, 0) + 1
                sessions.append(Session(tr, 0, "capycap", "human", f"capycap:{rec.get('index', f'{fp.stem}:{i}')}", {"game": gt}))
    if not sessions:
        raise ValueError("CaptchaSolve30k loader found no usable sessions (check --capycap path / run inspect_data).")
    print(f"[capycap] {len(sessions)} mouse sessions via '{source}' (skipped {skipped_touch} touch, "
          f"{skipped_short} too short/undecodable); games: {games}")
    return sessions


# ----------------------------------------------------------------- app-collected
def load_collected(path):
    """Sessions logged by the Flask app in COLLECT mode (collected/sessions.jsonl)."""
    out = []
    for i, line in enumerate(open(path)):
        r = json.loads(line)
        if r.get("mouse") and r.get("label") in ("human", "bot"):
            lab = int(r["label"] == "bot")
            out.append(Session(np.array(r["mouse"], float), lab, "collected",
                               "human" if not lab else r.get("bot_type", "bot"), f"collected:{i}",
                               {"pointer_type": r.get("pointer_type")}))
    return out
