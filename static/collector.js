/* Behaviour collector + challenge flow.
 * Privacy: records only pointer geometry/timing, key *intervals* (never which key),
 * and counts. No text content, no key codes, no persistent identifiers. */
(function () {
  "use strict";
  const T0 = performance.now();
  const S = { mouse: [], ptype: null, clicks: 0, keyT: [], scroll: 0, focus: 0, blur: 0, paste: 0,
              untrusted: 0, tFirst: null };
  const MAXP = 1500;
  const mark = () => { if (S.tFirst === null) S.tFirst = performance.now() - T0; };
  const chk = (e) => { if (e.isTrusted === false) S.untrusted++; };

  addEventListener("pointermove", (e) => {
    chk(e); S.ptype = e.pointerType || S.ptype;
    if (e.pointerType && e.pointerType !== "mouse") { mark(); return; }  // model is mouse-only
    S.mouse.push([e.clientX / innerWidth, e.clientY / innerHeight, +(e.timeStamp - T0).toFixed(1)]);
    if (S.mouse.length > MAXP) S.mouse.shift();
    mark();
  }, { passive: true });
  addEventListener("pointerdown", (e) => { chk(e); S.clicks++; S.ptype = S.ptype || e.pointerType; mark(); }, true);
  addEventListener("keydown", (e) => { chk(e); S.keyT.push(Math.round(e.timeStamp - T0)); if (S.keyT.length > 200) S.keyT.shift(); mark(); }, true);
  addEventListener("scroll", () => { S.scroll++; }, { passive: true });
  addEventListener("focus", () => { S.focus++; });
  addEventListener("blur", () => { S.blur++; });
  addEventListener("paste", () => { S.paste++; mark(); }, true);

  function snapshot() {
    return {
      mouse: S.mouse, pointer_type: S.ptype || "none", n_clicks: S.clicks, n_keys: S.keyT.length,
      key_intervals: S.keyT.slice(1).map((t, i) => t - S.keyT[i]).slice(-60),
      n_scroll: S.scroll, n_focus: S.focus, n_blur: S.blur, n_paste: S.paste, untrusted: S.untrusted,
      t_first_ms: S.tFirst, t_now_ms: Math.round(performance.now() - T0),
      webdriver: !!navigator.webdriver, viewport: [innerWidth, innerHeight],
    };
  }

  // ---- compact SHA-256 (sync) so proof-of-work works on non-HTTPS origins too ----
  const K = new Uint32Array([0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2]);
  const W32 = new Uint32Array(64);
  function sha256(str) {
    const msg = new TextEncoder().encode(str), l = msg.length;
    const n = ((l + 9 + 63) >> 6) << 6, buf = new Uint8Array(n);
    buf.set(msg); buf[l] = 0x80;
    const dv = new DataView(buf.buffer); dv.setUint32(n - 4, l * 8, false);
    let h0=0x6a09e667,h1=0xbb67ae85,h2=0x3c6ef372,h3=0xa54ff53a,h4=0x510e527f,h5=0x9b05688c,h6=0x1f83d9ab,h7=0x5be0cd19;
    for (let o = 0; o < n; o += 64) {
      for (let i = 0; i < 16; i++) W32[i] = dv.getUint32(o + i * 4, false);
      for (let i = 16; i < 64; i++) {
        const a = W32[i-15], b = W32[i-2];
        const s0 = ((a>>>7)|(a<<25)) ^ ((a>>>18)|(a<<14)) ^ (a>>>3);
        const s1 = ((b>>>17)|(b<<15)) ^ ((b>>>19)|(b<<13)) ^ (b>>>10);
        W32[i] = (W32[i-16] + s0 + W32[i-7] + s1) | 0;
      }
      let a=h0,b=h1,c=h2,d=h3,e=h4,f=h5,g=h6,h=h7;
      for (let i = 0; i < 64; i++) {
        const S1 = ((e>>>6)|(e<<26)) ^ ((e>>>11)|(e<<21)) ^ ((e>>>25)|(e<<7));
        const ch = (e & f) ^ (~e & g);
        const t1 = (h + S1 + ch + K[i] + W32[i]) | 0;
        const S0 = ((a>>>2)|(a<<30)) ^ ((a>>>13)|(a<<19)) ^ ((a>>>22)|(a<<10));
        const mj = (a & b) ^ (a & c) ^ (b & c);
        const t2 = (S0 + mj) | 0;
        h=g; g=f; f=e; e=(d + t1)|0; d=c; c=b; b=a; a=(t1 + t2)|0;
      }
      h0=(h0+a)|0;h1=(h1+b)|0;h2=(h2+c)|0;h3=(h3+d)|0;h4=(h4+e)|0;h5=(h5+f)|0;h6=(h6+g)|0;h7=(h7+h)|0;
    }
    return [h0,h1,h2,h3,h4,h5,h6,h7];
  }
  function lzb(words) {
    let n = 0;
    for (const w of words) { if (w === 0) { n += 32; continue; } n += Math.clz32(w); break; }
    return n;
  }
  async function solvePow(nonce, bits) {
    if (!bits) return 0;
    for (let c = 0; ; c++) {
      if (lzb(sha256(nonce + ":" + c)) >= bits) return c;
      if (c % 3000 === 2999) await new Promise((r) => setTimeout(r));   // keep UI responsive
    }
  }
  window.__sha256 = sha256; window.__lzb = lzb;   // exposed for tests

  // ---- UI flow: assess -> challenge -> login ----
  const $ = (id) => document.getElementById(id);
  let challenge = null, powCounter = 0;

  async function post(url, body) {
    const r = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    let j = {}; try { j = await r.json(); } catch (e) {}
    return { status: r.status, body: j };
  }
  function setMsg(text, ok) { const m = $("msg"); m.textContent = text || ""; m.className = ok ? "ok" : "err"; }
  function debug(res) {
    const d = $("debug"); if (!d || !res.debug) return;
    d.hidden = false;
    d.textContent = "tier: " + res.tier.toUpperCase() + "  risk: " + res.debug.risk.toFixed(3) +
      "  mode: " + res.debug.mode + "\n" + res.debug.reasons.join("\n");
  }
  function refreshCaptcha() { $("cap").src = "/captcha.png?" + Date.now(); $("capans").value = ""; }

  async function start() {
    const user = $("username").value.trim();
    if (!user) return setMsg("Please enter a username.");
    $("go").disabled = true; setMsg("Checking…", true);
    const snap = snapshot(); snap.honeypot = $("website").value.length > 0;
    snap._label = window.__LABEL || undefined; snap._bot_type = undefined;
    const r = await post("/api/assess", snap);
    if (r.status === 429) { $("go").disabled = false; return setMsg("Too many attempts. Try again in " + (r.body.retry_after || 30) + "s."); }
    if (r.status !== 200) { $("go").disabled = false; return setMsg("Could not start verification."); }
    challenge = r.body; debug(challenge);
    $("tierlabel").textContent = challenge.label;
    if (challenge.captcha_url) {
      $("capbox").hidden = false; $("cap").src = challenge.captcha_url + "?" + Date.now();
      $("go").hidden = true; $("submit").hidden = false; setMsg("");
      if (challenge.pow) { $("submit").disabled = true; powCounter = await solvePow(challenge.pow.nonce, challenge.pow.bits); $("submit").disabled = false; }
    } else {
      setMsg("Verifying…", true);
      powCounter = await solvePow(challenge.pow.nonce, challenge.pow.bits);
      await login();
    }
  }
  async function login() {
    const body = { username: $("username").value.trim(), captcha: $("capans").value, pow_counter: powCounter,
                   honeypot: $("website").value.length > 0 };
    const r = await post("/api/login", body), b = r.body;
    if (r.status === 200 && b.ok) { setMsg(b.message, true); $("submit").hidden = true; $("go").hidden = true; return; }
    setMsg(b.message || "Verification failed.");
    if (b.retry_after) setMsg((b.message || "Locked.") + " (" + b.retry_after + "s)");
    // start over so a fresh risk assessment (including this failure) is made
    $("capbox").hidden = true; $("submit").hidden = true; $("go").hidden = false; $("go").disabled = false;
  }
  window.addEventListener("DOMContentLoaded", () => {
    $("go").addEventListener("click", start);
    $("submit").addEventListener("click", login);
    $("refresh").addEventListener("click", (e) => { e.preventDefault(); refreshCaptcha(); });
  });
})();
