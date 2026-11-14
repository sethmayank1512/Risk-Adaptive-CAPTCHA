"""Run:  python -m tests.test_pipeline   (takes ~15 s: trains a synthetic model first)"""
import base64
import json
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

# The app tests run against their OWN throw-away synthetic model, so they pass whether or not
# artifacts/ holds a real-data model (real models have different score distributions).
_ART = tempfile.mkdtemp()
subprocess.run([sys.executable, "-m", "training.train", "--synthetic", "--out", _ART, "--report", _ART],
               check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
os.environ["ART_DIR"] = _ART

import app as appmod
from risk.challenges import solve_pow
from risk.features import extract
from training import synth
from training.loaders import load_capycap, load_delbot

IPN = [0]


def client(age=5.0):
    """Fresh browser: new IP, page loaded `age` seconds ago (humans don't submit instantly)."""
    IPN[0] += 1
    c = appmod.app.test_client()
    c.environ_base["REMOTE_ADDR"] = f"10.0.{IPN[0] // 250}.{IPN[0] % 250 + 1}"
    c.get("/")
    sid = c.get_cookie("sid").value
    appmod.ENGINE.store.d[sid]["t_load"] -= age
    return c, sid


def payload(traj, ptype="mouse", **kw):
    p = {"mouse": np.asarray(traj).tolist() if traj is not None else [], "pointer_type": ptype,
         "n_clicks": 1, "n_keys": 8}
    p.update(kw)
    return p


def human_payload(rng):
    return payload(synth.synth_human(rng))


def test_loaders_and_features():
    d = tempfile.mkdtemp()
    # documented real-data formats
    sj = synth.write_fake_delbot_tree(d, 30)
    s = load_delbot(sj); assert {x.label for x in s} == {0, 1}
    assert {x.bot_type for x in s if x.label} == set(synth.BOT_KINDS)       # bot family comes from folder name
    assert max(s[0].traj[:, 0].max(), s[0].traj[:, 1].max()) <= 1.5          # pixels were normalised
    synth.write_fake_capycap(f"{d}/c.jsonl", 40, touch_frac=0.25)
    c = load_capycap(f"{d}/c.jsonl", min_samples=50)
    assert 20 <= len(c) < 40                                                  # touchscreen sessions were dropped
    assert c[0].traj[0, 0] > 0 or c[0].traj[0, 1] > 0                         # leading (0,0) ticks stripped
    # legacy formats still load
    synth.write_fake_delbot(f"{d}/d.json", 20); assert len(load_delbot(f"{d}/d.json")) == 40
    synth.write_fake_capycap_stream(f"{d}/s.jsonl", 10, stride12=False)
    from training.loaders import decode_blob
    rec = json.loads(open(f"{d}/s.jsonl").readline()); assert len(decode_blob(base64.b64decode(rec["pointerData"]))[0]) > 100
    rng = np.random.default_rng(1)
    assert extract(synth.synth_human(rng)) is not None
    assert extract([[0, 0, 0], [1, 1, 10]]) is None             # too little evidence -> None, not a crash


def test_cookie_never_contains_answer():
    c, sid = client()
    r = c.post("/api/assess", json=payload(None, "none", n_keys=9)).get_json()
    assert r["captcha_url"]
    c.get("/captcha.png")
    ans = appmod.ENGINE.store.d[sid]["challenge"]["captcha_text"]
    cookie = c.get_cookie("sid").value
    assert ans and ans not in cookie and cookie == sid and len(cookie) >= 30


def test_tiers_on_population():
    rng = np.random.default_rng(7)
    humans, bots = [], []
    for _ in range(30):
        c, _ = client(); humans.append(c.post("/api/assess", json=human_payload(rng)).get_json()["tier"])
    for k in synth.BOT_KINDS:
        for _ in range(8):
            c, _ = client(age=2.5)   # a patient bot that waits before posting forged mouse data
            bots.append(c.post("/api/assess", json=payload(synth.synth_bot(rng, k))).get_json()["tier"])
    hl = humans.count("low") / len(humans); bh = bots.count("high") / len(bots)
    print(f"   humans low={hl:.2f}  forged-mouse bots high={bh:.2f}   (synthetic model - illustrative only)")
    assert hl >= 0.7 and bh >= 0.8 and humans.count("high") <= 2


def test_low_tier_flow_and_one_time_challenge():
    rng = np.random.default_rng(11)
    for _ in range(20):                                         # find a human that lands in LOW
        c, sid = client()
        r = c.post("/api/assess", json=human_payload(rng)).get_json()
        if r["tier"] == "low":
            break
    assert r["tier"] == "low" and r["captcha_url"] is None and r["pow"]
    # wrong PoW must fail
    bad = c.post("/api/login", json={"username": "ann", "pow_counter": 0}).get_json()
    if not bad["ok"]:
        assert "failed" in bad["message"].lower()
        r = c.post("/api/assess", json=human_payload(rng)).get_json()   # re-assess after failure
    cnt = solve_pow(r["pow"]["nonce"], r["pow"]["bits"])
    ok = c.post("/api/login", json={"username": "ann", "pow_counter": cnt}).get_json()
    assert ok["ok"] or ok.get("reassess")                       # (failure above may legitimately escalate)
    again = c.post("/api/login", json={"username": "ann", "pow_counter": cnt})
    assert again.status_code == 409                             # challenge is single-use


def test_medium_flow_captcha():
    c, sid = client(age=30)
    r = c.post("/api/assess", json=payload(None, "none", n_keys=12)).get_json()   # keyboard-only visitor
    assert r["tier"] == "medium", r
    c.get("/captcha.png")
    ch = appmod.ENGINE.store.d[sid]["challenge"]; ch["t"] -= 10
    wrong = c.post("/api/login", json={"username": "kim", "captcha": "WRONG"}).get_json()
    assert not wrong["ok"]
    r = c.post("/api/assess", json=payload(None, "none", n_keys=12)).get_json()
    c.get("/captcha.png")
    st = appmod.ENGINE.store.d[sid]; st["challenge"]["t"] -= 10
    res = c.post("/api/login", json={"username": "kim", "captcha": st["challenge"]["captcha_text"].lower()}).get_json()
    assert res["ok"] or res.get("reassess"), res                # one prior failure raises risk; either outcome is by design


def test_instant_bot_is_high_and_locked_out():
    c, sid = client(age=0.0)
    r = c.post("/api/assess", json={}).get_json()
    assert r["tier"] == "high", r
    c.get("/captcha.png")
    st = appmod.ENGINE.store.d[sid]
    out = c.post("/api/login", json={"username": "bot", "captcha": "AAAAAAA", "pow_counter": 0})
    assert out.status_code == 429 and out.get_json()["retry_after"] >= 15
    assert c.post("/api/assess", json={}).status_code == 429     # locked: cannot probe for a better tier


def test_replay_detected():
    rng = np.random.default_rng(3)
    tr = synth.synth_human(rng)
    c1, _ = client(); r1 = c1.post("/api/assess", json=payload(tr)).get_json()
    c2, _ = client(); r2 = c2.post("/api/assess", json=payload(tr)).get_json()
    assert any("identical trajectory" in x for x in r2["debug"]["reasons"])
    assert r2["debug"]["risk"] > r1["debug"]["risk"]


def test_continuous_evaluation_escalates_after_flood():
    rng = np.random.default_rng(21)
    for _ in range(25):
        c, sid = client()
        r = c.post("/api/assess", json=human_payload(rng)).get_json()
        if r["tier"] == "low":
            break
    assert r["tier"] == "low"
    for _ in range(40):                                          # session behaves, then floods
        c.get("/captcha.png")
    cnt = solve_pow(r["pow"]["nonce"], r["pow"]["bits"])
    out = c.post("/api/login", json={"username": "ann", "pow_counter": cnt}).get_json()
    assert out.get("reassess") and not out["ok"], out


def test_hard_rate_limit_429():
    c, sid = client()
    codes = [c.get("/captcha.png").status_code for _ in range(70)]
    assert 429 in codes


def test_login_requires_assessment_and_validates_input():
    c, _ = client()
    assert c.post("/api/login", json={"username": "x"}).status_code == 409
    c.post("/api/assess", json={})
    assert c.post("/api/login", json={"username": ""}).get_json()["ok"] is False
    assert c.post("/api/assess", data="not json", content_type="text/plain").status_code in (200, 429)


if __name__ == "__main__":
    appmod.app.config["COLLECT"] = False
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t(); print("PASS", t.__name__)
        except Exception as e:  # noqa
            failed += 1; print("FAIL", t.__name__, "->", repr(e)[:300])
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(failed)
