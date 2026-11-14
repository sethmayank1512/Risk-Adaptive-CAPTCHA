"""HTTP-level scripted bots of increasing sophistication, to test the *deployed* app.

    python -m tools.bots --url http://127.0.0.1:5000 --n 20
    # with the server started as CAPTCHA_COLLECT=1, add --label so sessions are logged for training/eval:
    python -m tools.bots --n 30 --label

These bots never see a browser; they post fabricated behaviour straight to the API,
which is exactly the 'forged client features' threat. Each bot only measures the TIER
it is given (it does not try to solve the CAPTCHA), so the output answers:
"how often does each kind of bot get the cheap path?"
"""
import argparse
import collections
import time

import numpy as np
import requests

from training import synth


def run_bot(url, kind, rng, label):
    s = requests.Session()
    s.get(url + "/")
    payload = {"pointer_type": "none"}
    if kind == "naive_no_events":
        pass                                                      # posts immediately, no behaviour at all
    elif kind == "webdriver_straight":
        time.sleep(0.3)
        payload = {"pointer_type": "mouse", "mouse": synth.synth_bot(rng, "naive_dense").tolist(), "n_clicks": 1, "webdriver": True}
    elif kind == "forged_bezier":
        time.sleep(1.5)
        payload = {"pointer_type": "mouse", "mouse": synth.synth_bot(rng, "bezier").tolist(), "n_clicks": 1}
    elif kind == "forged_noisy_bezier":
        time.sleep(2.0)
        payload = {"pointer_type": "mouse", "mouse": synth.synth_bot(rng, "noisy_bezier").tolist(), "n_clicks": 1}
    elif kind == "replay_of_human":
        time.sleep(2.0)
        payload = {"pointer_type": "mouse", "mouse": synth.synth_replay(rng, synth.synth_human(rng)).tolist(), "n_clicks": 1}
    elif kind == "flood":
        for _ in range(25):                                       # hammer the API first
            s.post(url + "/api/assess", json={})
        time.sleep(1.0)
        payload = {"pointer_type": "mouse", "mouse": synth.synth_bot(rng, "noisy_bezier").tolist(), "n_clicks": 1}
    if label:
        payload.update(_label="bot", _bot_type=kind)
    r = s.post(url + "/api/assess", json=payload)
    if r.status_code != 200:
        return "blocked(%d)" % r.status_code, None
    j = r.json()
    return j["tier"], j.get("debug", {}).get("risk")


KINDS = ["naive_no_events", "webdriver_straight", "forged_bezier", "forged_noisy_bezier", "replay_of_human", "flood"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:5000")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--label", action="store_true", help="log sessions (server needs CAPTCHA_COLLECT=1)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    print(f"{'bot kind':24s} {'low':>5s} {'medium':>7s} {'high':>5s} {'blocked':>8s}   (share of {a.n})")
    for k in KINDS:
        c = collections.Counter(run_bot(a.url, k, rng, a.label)[0] for _ in range(a.n))
        blocked = sum(v for t, v in c.items() if t.startswith("blocked"))
        print(f"{k:24s} {c['low']/a.n:5.2f} {c['medium']/a.n:7.2f} {c['high']/a.n:5.2f} {blocked/a.n:8.2f}")
        time.sleep(1)


if __name__ == "__main__":
    main()
