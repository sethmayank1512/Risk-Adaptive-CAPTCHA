"""Risk-adaptive CAPTCHA web app.

Flow:  page load -> user clicks Continue -> POST /api/assess (behaviour snapshot)
       -> server picks a tier and issues a challenge -> POST /api/login (answer)
       -> server RE-EVALUATES risk (rate, failures, ...) before accepting.

Env vars:  CAPTCHA_DEBUG=1 (default) shows the risk breakdown in the UI
           CAPTCHA_COLLECT=1 logs labelled sessions to collected/ (for your own test set)
           ART_DIR=artifacts   HARD_LIMIT=60 (requests/min/IP before HTTP 429)
"""
import io
import json
import os
import time
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_file

from risk.challenges import TIER_CFG, make_pow, new_text, render_captcha, verify_pow
from risk.engine import TIER_RANK, Engine
from risk.model import TrajModel

app = Flask(__name__)
app.config.update(
    DEBUG_PANEL=os.environ.get("CAPTCHA_DEBUG", "1") == "1",
    COLLECT=os.environ.get("CAPTCHA_COLLECT", "0") == "1",
    HARD_LIMIT=int(os.environ.get("HARD_LIMIT", "60")),
    COLLECT_DIR=Path(os.environ.get("COLLECT_DIR", "collected")),
)
MODEL = TrajModel(os.environ.get("ART_DIR", "artifacts"))
ENGINE = Engine(MODEL)
CHALLENGE_TTL = 180


# ------------------------------------------------------------------ session plumbing
def client_ip():
    return request.remote_addr or "0.0.0.0"   # behind a proxy: use werkzeug ProxyFix


def load_state():
    """Opaque 'sid' cookie -> server-side state. Anything unknown gets a fresh state."""
    sid = request.cookies.get("sid")
    st = ENGINE.store.get(sid) if sid else None
    if st is None:
        sid, st = ENGINE.store.create()
        request.environ["new_sid"] = sid
    return sid, st


@app.before_request
def throttle():
    if request.path.startswith("/static"):
        return None
    sid, st = load_state()
    request.environ["sid"], request.environ["st"] = sid, st
    ip = client_ip()
    ENGINE.limiter.hit(("ip", ip))
    ENGINE.limiter.hit(("sid", sid))
    if ENGINE.limiter.count(("ip", ip), 60) > app.config["HARD_LIMIT"]:        # hard rate limit
        resp = jsonify(error="rate_limited", message="Too many requests.", retry_after=30)
        resp.status_code, resp.headers["Retry-After"] = 429, "30"
        return resp
    return None


@app.after_request
def set_cookie(resp):
    sid = request.environ.get("new_sid")
    if sid:
        resp.set_cookie("sid", sid, httponly=True, samesite="Lax", max_age=3600)
    resp.headers["Cache-Control"] = "no-store"
    return resp


def locked_response(st, now):
    if st["locked_until"] > now:
        wait = int(st["locked_until"] - now) + 1
        resp = jsonify(ok=False, error="locked", message=f"Too many failed attempts. Try again in {wait}s.", retry_after=wait)
        resp.status_code, resp.headers["Retry-After"] = 429, str(wait)
        return resp
    return None


# ------------------------------------------------------------------ routes
@app.route("/")
def index():
    st = request.environ["st"]
    st["t_load"], st["challenge"] = time.time(), None
    return render_template("index.html", banner=MODEL.banner(),
                           collect_label=("human" if app.config["COLLECT"] else None))


@app.route("/api/assess", methods=["POST"])
def assess():
    sid, st, ip, now = request.environ["sid"], request.environ["st"], client_ip(), time.time()
    if (r := locked_response(st, now)):
        return r
    payload = request.get_json(silent=True)
    payload = payload if isinstance(payload, dict) else {}
    res = ENGINE.assess(st, sid, ip, payload, now)
    tier, cfg = res["tier"], TIER_CFG[res["tier"]]
    ch = {"tier": tier, "t": now, "used": False, "captcha_text": None,
          "pow": make_pow(cfg["pow_bits"]) if cfg["pow_bits"] else None}
    st["challenge"] = ch
    out = {"tier": tier, "label": cfg["label"], "pow": ch["pow"],
           "captcha_url": "/captcha.png" if cfg["captcha"] else None}
    if app.config["DEBUG_PANEL"]:
        out["debug"] = {"risk": res["risk"], "mode": res["mode"], "reasons": res["reasons"]}
    if app.config["COLLECT"] and payload.get("_label") in ("human", "bot"):
        log_collected(payload, res, ip)
    return jsonify(out)


@app.route("/captcha.png")
def captcha_image():
    st = request.environ["st"]
    ch = st.get("challenge")
    if not ch or ch["used"] or not TIER_CFG[ch["tier"]]["captcha"]:
        return ("", 404)
    cfg = TIER_CFG[ch["tier"]]
    ch["captcha_text"] = new_text(cfg["length"])        # (re)generated on every fetch = "refresh"
    buf = io.BytesIO()
    render_captcha(ch["captcha_text"], cfg["captcha"]).save(buf, format="PNG")
    buf.seek(0)
    return send_file(buf, mimetype="image/png")


@app.route("/api/login", methods=["POST"])
def login():
    sid, st, ip, now = request.environ["sid"], request.environ["st"], client_ip(), time.time()
    if (r := locked_response(st, now)):
        return r
    ch = st.get("challenge")
    if not ch or ch["used"] or not st.get("assessment"):
        return jsonify(ok=False, error="assess_first", message="Please start verification again."), 409
    data = request.get_json(silent=True)
    data = data if isinstance(data, dict) else {}
    username = str(data.get("username", "")).strip()[:64]
    if not username or not username.isprintable():
        return jsonify(ok=False, message="Please enter a username.")     # input error, not penalised

    ch["used"] = True                                                    # one-time challenge
    now_tier, risk, why = ENGINE.reevaluate(st, sid, ip, now)            # continuous evaluation
    if TIER_RANK[now_tier] > TIER_RANK[ch["tier"]]:
        st["challenge"] = None
        return jsonify(ok=False, reassess=True, message="Additional verification is required.",
                       **({"debug_reasons": why} if app.config["DEBUG_PANEL"] else {}))

    cfg = TIER_CFG[ch["tier"]]
    problems = []
    if data.get("honeypot"):
        problems.append("honeypot")
    if now - ch["t"] > CHALLENGE_TTL:
        problems.append("expired")
    if ch["pow"] and not verify_pow(ch["pow"]["nonce"], data.get("pow_counter"), ch["pow"]["bits"]):
        problems.append("pow")
    if cfg["captcha"]:
        ans = str(data.get("captcha", "")).strip().upper()
        if not ch["captcha_text"] or ans != ch["captcha_text"]:
            problems.append("captcha")
    if now - ch["t"] < cfg["min_wait"]:
        problems.append("too_fast")

    if problems:
        st["fails"] += 1
        ENGINE.limiter.hit(("ipfail", ip), now)
        st["challenge"] = None
        msg = {"captcha": "Incorrect CAPTCHA. Please try again.", "too_fast": "That was too fast. Please try again."
               }.get(problems[0], "Verification failed. Please try again.")
        resp = {"ok": False, "message": msg}
        if st["fails"] >= cfg["max_attempts"] or ch["tier"] == "high":    # lockout with exponential back-off
            lock = min(300, 15 * 2 ** max(0, st["fails"] - 1))
            st["locked_until"] = now + lock
            resp.update(retry_after=lock, message=f"{msg} Locked for {lock}s.")
        return jsonify(resp), (429 if "retry_after" in resp else 200)

    st["fails"] = 0
    st["challenge"] = None
    return jsonify(ok=True, message=f"Verified ({ch['tier']} risk path). Welcome, {username}!")


# ------------------------------------------------------------------ data collection (optional)
def log_collected(payload, res, ip):
    d = app.config["COLLECT_DIR"]; d.mkdir(exist_ok=True)
    rec = {"ts": time.time(), "label": payload["_label"], "bot_type": str(payload.get("_bot_type") or "bot")[:30],
           "pointer_type": res["mode"], "tier": res["tier"], "risk": round(res["risk"], 4),
           "mouse": res["mouse"].round(5).tolist() if res["mouse"] is not None else None}
    with open(d / "sessions.jsonl", "a") as fh:
        fh.write(json.dumps(rec) + "\n")


if __name__ == "__main__":
    app.run(debug=False, port=5000)
