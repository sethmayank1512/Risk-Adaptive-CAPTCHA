"""Print what is actually inside the dataset files before training.
    python -m training.inspect_data --delbot data/DELBOT-Mouse/python/sessions.json \
                                    --capycap data/CaptchaSolve30k/validation.jsonl
"""
import argparse
import collections
import json
from pathlib import Path

import numpy as np

from training.loaders import _ticks_to_traj, load_delbot, parse_text_trajectory


def inspect_delbot(path):
    p = Path(path)
    jp = p if p.is_file() else next(iter(sorted(p.rglob("sessions.json"))), None)
    if jp is None:
        print("No sessions.json found under", path); return
    d = json.load(open(jp))
    print(f"{jp}: " + ", ".join(f"{k}={len(v)} entries" for k, v in d.items() if isinstance(v, list)))
    base = jp.parent
    for key in ("human", "bot"):
        items = d.get(key) or []
        if not items:
            continue
        ex = items[0]
        print(f"\n[{key}] first entry: {ex[:150]!r}")
        found = sum(1 for r in items[:200] if (base / r).is_file() or (base.parent / r).is_file())
        print(f"[{key}] {found}/{min(len(items), 200)} of the first 200 paths exist on disk")
        fp = next((q for q in (base / ex, base.parent / ex) if q.is_file()), None)
        if fp is None:
            print(f"   -> files missing. Extract the dataset first:\n      cd {base} && tar -xzf dataset.tar.gz")
            continue
        txt = fp.read_text(errors="ignore")
        print(f"[{key}] {fp} ({len(txt.splitlines())} lines). RAW FIRST 6 LINES:")
        for ln in txt.splitlines()[:6]:
            print("     ", repr(ln[:140]))
        pts, layout = parse_text_trajectory(txt)
        print(f"[{key}] parsed: {None if pts is None else len(pts)} points, layout={layout}")
    try:
        s = load_delbot(path)
        print("\nLoader result:", len(s), "sessions;", dict(collections.Counter(x.bot_type for x in s)))
        print("Median points/session:", sorted(len(x.traj) for x in s)[len(s) // 2])
        print("CHECK: bot families above come from folder names; are they sensible?")
    except Exception as e:
        print("\nLoader failed:", e)


def inspect_capycap(path):
    p = Path(path)
    f = p if p.is_file() else next(p.glob("*.jsonl"))
    touch = collections.Counter(); games = collections.Counter(); ok = 0; durs = []; lead = []; n = 0
    with open(f) as fh:
        for i, line in enumerate(fh):
            if i >= 500:
                break
            r = json.loads(line); n += 1
            if i == 0:
                print(f"--- first record of {f} ---")
                for k, v in r.items():
                    print(f"  {k!r}: {type(v).__name__}" + (f" len={len(v)}" if hasattr(v, '__len__') else f" = {v}"))
                t0 = r["tickInputs"][0] if isinstance(r.get("tickInputs"), list) and r["tickInputs"] else None
                print("  tickInputs[0] =", t0)
            touch[r.get("touchscreen")] += 1; games[r.get("gameType")] += 1
            tr = _ticks_to_traj(r)
            if tr is not None:
                ok += 1; durs.append(tr[-1, 2] / 1000); lead.append(len(r["tickInputs"]) - len(tr))
    print(f"\nFirst {n} records: touchscreen={dict(touch)}; games={dict(games)}")
    print(f"tickInputs usable in {ok}/{n}; median live duration {np.median(durs):.1f}s; "
          f"median leading (0,0) ticks stripped: {int(np.median(lead))}")
    print("The loader uses `tickInputs` (documented, 240 Hz, 200x200 grid) and skips touchscreen sessions.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delbot"); ap.add_argument("--capycap")
    a = ap.parse_args()
    if a.delbot:
        inspect_delbot(a.delbot)
    if a.capycap:
        inspect_capycap(a.capycap)


if __name__ == "__main__":
    main()
