"""Risk engine: behaviour model + server-side signals -> risk score -> tier.

risk = sigmoid( logit(p_trajectory) + sum(server-side adjustments) )

* With no server-side red flags the adjustment is 0, so risk == the calibrated
  trajectory probability and the tier thresholds learnt in training apply as-is.
* If there is no usable mouse trajectory (touch, keyboard-only, screen reader,
  autofill, or a bot that sends nothing) the model is NOT applied: we start from
  a neutral 'unknown' prior that lands in the MEDIUM tier. Absence of mouse
  data is not evidence of a bot, but it is not evidence of a human either.
* Client-sent data is advisory. Rate, failures, dwell time and duplicates are
  measured by the server, and every login re-evaluates the risk (continuous
  evaluation), so a session that looks fine and then floods the server escalates.
"""
import hashlib
import math
import secrets
import threading
import time
from collections import defaultdict, deque

import numpy as np

from risk.features import MIN_POINTS, extract, last_window

TIER_RANK = {"low": 0, "medium": 1, "high": 2}
MAX_POINTS = 2000


def logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


class Store:
    """Server-side session state. The cookie only carries an opaque id, so nothing
    sensitive (e.g. the CAPTCHA answer) is ever readable by the client.
    In production replace with Redis."""

    def __init__(self, ttl=3600, max_items=20000):
        self.d, self.ttl, self.max, self.lock = {}, ttl, max_items, threading.Lock()

    def get(self, sid):
        with self.lock:
            st = self.d.get(sid)
            if st and time.time() - st["seen"] > self.ttl:
                del self.d[sid]
                return None
            if st:
                st["seen"] = time.time()
            return st

    def create(self):
        with self.lock:
            if len(self.d) > self.max:
                cutoff = time.time() - self.ttl
                self.d = {k: v for k, v in self.d.items() if v["seen"] > cutoff}
            sid = secrets.token_urlsafe(24)
            self.d[sid] = {"seen": time.time(), "t_load": time.time(), "fails": 0, "assess_n": 0,
                           "locked_until": 0.0, "challenge": None, "assessment": None, "sticky": 0.0}
            return sid, self.d[sid]


class RateLimiter:
    def __init__(self, horizon=900):
        self.ev, self.h, self.lock = defaultdict(deque), horizon, threading.Lock()

    def hit(self, key, now=None):
        now = now or time.time()
        with self.lock:
            q = self.ev[key]
            q.append(now)
            while q and q[0] < now - self.h:
                q.popleft()

    def count(self, key, window=60, now=None):
        now = now or time.time()
        with self.lock:
            q = self.ev.get(key)
            return sum(1 for t in q if t >= now - window) if q else 0


class Engine:
    def __init__(self, model, store=None, limiter=None):
        self.model = model
        self.store = store or Store()
        self.limiter = limiter or RateLimiter()
        self.seen_hashes = deque(maxlen=5000)   # for exact-replay detection
        self.z_unknown = (logit(model.low) + logit(model.high)) / 2

    # ---- helpers -------------------------------------------------------
    def tier_of(self, risk):
        return "low" if risk < self.model.low else ("high" if risk >= self.model.high else "medium")

    @staticmethod
    def sanitize_mouse(raw):
        if not isinstance(raw, list):
            return None
        pts = []
        for p in raw[-MAX_POINTS:]:
            if isinstance(p, (list, tuple)) and len(p) >= 3:
                try:
                    x, y, t = float(p[0]), float(p[1]), float(p[2])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(x) and math.isfinite(y) and math.isfinite(t):
                    pts.append((min(max(x, -0.5), 1.5), min(max(y, -0.5), 1.5), t))
        return np.array(pts) if len(pts) >= 2 else None

    def _dup(self, mouse):
        """Exact-replay check: hash of the (quantised) head of the trajectory."""
        if mouse is None or len(mouse) < MIN_POINTS:
            return False
        q = np.round(mouse[:40, :2] * 2000).astype(int).tobytes() + np.round(np.diff(mouse[:40, 2]) / 4).astype(int).tobytes()
        h = hashlib.sha1(q).hexdigest()
        dup = h in self.seen_hashes
        self.seen_hashes.append(h)
        return dup

    def server_adjustments(self, st, ip, now, sig):
        """Additive logit adjustments from server-measured signals. Returns (total, reasons)."""
        adj, why = 0.0, []
        def add(v, label):
            nonlocal adj
            if v:
                adj += v
                why.append(f"{label}: {v:+.2f}")
        if sig.get("honeypot"):
            add(10.0, "honeypot filled")
        if sig.get("webdriver"):
            add(1.5, "navigator.webdriver set")
        if sig.get("untrusted_events", 0) > 0:
            add(1.5, "synthetic (untrusted) DOM events")
        dwell = sig.get("dwell_s")
        if dwell is not None:
            add(2.0 if dwell < 0.3 else 1.0 if dwell < 1.0 else 0.0, f"submitted {dwell:.2f}s after page load")
        ip_rate = self.limiter.count(("ip", ip), 60, now)
        sid_rate = self.limiter.count(("sid", sig.get("sid")), 60, now)
        add(min(3.0, 0.15 * (ip_rate - 10)) if ip_rate > 10 else 0.0, f"{ip_rate} req/min from IP")
        add(min(3.0, 0.15 * (sid_rate - 12)) if sid_rate > 12 else 0.0, f"{sid_rate} req/min in session")
        ip_fails = self.limiter.count(("ipfail", ip), 600, now)
        add(min(3.0, 0.7 * st["fails"] + 0.3 * ip_fails) if (st["fails"] or ip_fails) else 0.0,
            f"{st['fails']} failed attempts (+{ip_fails} from IP)")
        add(min(3.0, 0.5 * max(0, st["assess_n"] - 2)) if st["assess_n"] > 2 else 0.0, "repeated re-assessment (probing)")
        if sig.get("dup"):
            add(3.0, "identical trajectory seen before (replay)")
        if sig.get("no_interaction"):
            add(1.0, "no interaction events at all")
        add(st["sticky"], "sticky escalation") if st["sticky"] > 0 else None
        st["sticky"] = max(st["sticky"], min(2.0, 0.5 * max(0.0, adj - 1.0)))   # escalation never relaxes within a session
        return adj, why

    # ---- assessment ------------------------------------------------------
    def assess(self, st, sid, ip, payload, now=None):
        now = now or time.time()
        st["assess_n"] += 1
        mouse = self.sanitize_mouse(payload.get("mouse"))
        ptype = str(payload.get("pointer_type") or "none")[:10]
        feats, p_traj, mode = None, None, ptype
        if ptype == "mouse" and mouse is not None:
            win = last_window(mouse, 6.0)
            feats = extract(win) if win is not None else None
            if feats is not None:
                p_traj = self.model.score(feats)
                mode = "mouse"
            else:
                mode = "mouse-insufficient"
        z_base = logit(p_traj) if p_traj is not None else self.z_unknown
        sig = {
            "sid": sid,
            "webdriver": bool(payload.get("webdriver")),
            "untrusted_events": int(payload.get("untrusted") or 0),
            "honeypot": bool(payload.get("honeypot")),
            "dwell_s": max(0.0, now - st["t_load"]),
            "dup": self._dup(mouse) if mode == "mouse" else False,
            "no_interaction": (ptype == "none" and not payload.get("n_keys") and not payload.get("n_clicks")),
        }
        adj, why = self.server_adjustments(st, ip, now, sig)
        risk = sigmoid(z_base + adj)
        st["assessment"] = {"z_base": z_base, "t": now, "sig": {k: v for k, v in sig.items() if k in ("sid", "webdriver", "untrusted_events", "no_interaction")}}
        return {"tier": self.tier_of(risk), "risk": risk, "p_traj": p_traj, "mode": mode,
                "reasons": ([f"model p(bot)={p_traj:.3f}"] if p_traj is not None else ["no usable mouse trajectory -> neutral prior"]) + why,
                "mouse": mouse, "feats": feats}

    def reevaluate(self, st, sid, ip, now=None):
        """Continuous evaluation: re-score with fresh server-side signals at every login attempt."""
        now = now or time.time()
        a = st["assessment"]
        sig = dict(a["sig"]); sig["sid"] = sid
        adj, why = self.server_adjustments(st, ip, now, sig)
        risk = sigmoid(a["z_base"] + adj)
        return self.tier_of(risk), risk, why
