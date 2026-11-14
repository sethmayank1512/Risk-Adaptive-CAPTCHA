# Risk-adaptive CAPTCHA (Flask + behavioural risk model)

Estimate risk first, then apply an appropriate level of verification.

```
 browser (static/collector.js)                      server (app.py + risk/)
 ─────────────────────────────                      ──────────────────────────────────────────
 mouse trajectory, key *intervals*,  ──/api/assess─▶ features.extract ─▶ RandomForest (calibrated)
 clicks, scroll, focus, honeypot                     + server-side signals (rate, failures, dwell,
                                                      duplicates, webdriver/untrusted events)
                                                     ─▶ risk ─▶ tier ─┬─ LOW    PoW(12 bit) + honeypot, no visible challenge
                                                                      ├─ MEDIUM 5-char text CAPTCHA
 answer + PoW counter  ───────────/api/login───────▶ RE-EVALUATE risk └─ HIGH   7-char warped CAPTCHA + PoW(18 bit)
                                                      (continuous evaluation) + min-wait + lockout w/ back-off
```

## Quick start
```bash
pip install -r requirements.txt

# 0. optional smoke tests (use FAKE data; the app tests train their own throw-away model, ~15 s)
python -m training.train --synthetic --out /tmp/demo_art --report /tmp/demo_rep
python -m tests.test_pipeline

# 1. get the real data
git clone https://github.com/chrisgdt/DELBOT-Mouse data/DELBOT-Mouse
cd data/DELBOT-Mouse/python && tar -xzf dataset.tar.gz && cd ../../..   # sessions.json only lists paths; the trajectories are in this archive
pip install huggingface_hub
hf download Capycap-AI/CaptchaSolve30k --repo-type dataset --local-dir data/CaptchaSolve30k --include "validation.jsonl"   # ~800 MB

# 2. LOOK at the data (prints raw first lines + what the loaders detected)
python -m training.inspect_data --delbot data/DELBOT-Mouse/python/sessions.json \
                                --capycap data/CaptchaSolve30k/validation.jsonl

# 3. train + evaluate -> artifacts/ and reports/report.md (+ png plots)
python -m training.train --delbot data/DELBOT-Mouse/python/sessions.json \
                         --capycap data/CaptchaSolve30k/validation.jsonl

# 4. run the app, open http://127.0.0.1:5000
python app.py
```

### Data formats the loaders expect
* **DELBOT-Mouse**: `python/sessions.json` = `{"human": [relative paths], "bot": [relative paths]}`; each path is a text
  trajectory file inside the extracted `dataset.tar.gz`. The README does not document the file layout, so the parser
  auto-detects it (`t x y`, `x y t`, CSV, `key:value`, JSON; non-"Move" events are dropped). If `inspect_data`
  shows the wrong column order, pass `--delbot-layout txy` or `xyt`. The bot family (used for leave-one-bot-type-out)
  is taken from the folder name of each file. Pixel coordinates are normalised by a global screen estimate (`--screen`).
* **CaptchaSolve30k**: uses the documented `tickInputs` (240 Hz, x/y on a 200x200 grid), strips the leading (0,0)
  placeholder ticks and **skips `touchscreen: true` sessions** (the model is mouse-only). The binary `inputStream`
  is ignored: in the file we saw it is 4 bytes/sample, whereas the dataset card says 9, so the card looks out of date
  for that field (`--capycap-source stream` is a best-effort fallback).
* **Terms of use of CaptchaSolve30k** (dataset card): bot-detection research is explicitly permitted; re-identification,
  biometric profiling and behavioural *authentication of individuals* are not. This project classifies sessions as
  human/bot only. Keep the same restrictions if you redistribute data or derivatives (this repo ships no data).
Check the "Label sanity" printout in the training log: bots should be straighter / less jittery than humans.

## How the two datasets are used (and why not "bots from one, humans from the other")
* **DELBOT** supplies the labelled classes: humans and bots were recorded the same way, so differences
  reflect behaviour, not the data source.
* **CaptchaSolve30k** supplies *extra humans*, used for (a) an out-of-distribution false-positive test and
  (b) the optional `--final mixed` model. The report includes a **source-leakage check** (can the features
  tell DELBOT humans from CaptchaSolve humans?). If that AUC is high, do not mix sources.
* Features are computed on a **fixed 50 Hz resample** of random 1.5–6 s windows, so 240 Hz game ticks and
  browser mouse events become comparable. Raw-timing features exist but are excluded from the model.
* Splits are **by session/group** (never by window); thresholds come from held-out human scores.

## What the report gives you (`reports/report.md`)
Model comparison (LogReg / RF / HistGB: AUC, AP, TPR at 1% / 5% FPR) · source-leakage AUC ·
leave-one-bot-type-out detection · **where held-out humans land (usability cost)**, split by source ·
**where each bot family lands (security)**, including `syn_replay` which is never trained on ·
ROC/PR, score histogram, feature importances.

## Own deployment-realism test set
```bash
CAPTCHA_COLLECT=1 python app.py                      # humans: just use the page (label = human)
python -m tools.bots --n 30 --label                  # scripted bots log themselves as bots
python -m tools.summarize_collected collected/sessions.jsonl     # tier mix per human/bot kind
python -m training.train ... --collected collected/sessions.jsonl   # score them with the new model
```
Ask classmates to use phones, keyboard-only, and password managers too, and report their tier mix separately.

## Design decisions (answering the earlier critique)
| Concern | What the code does |
|---|---|
| Forgeable client features | Tier is decided and stored **server-side**; `/api/login` demands the challenge for that tier. Server-measured rate, failures, dwell time and duplicate trajectories feed the risk. Cookie is an opaque id (the original app's signed Flask cookie exposed the CAPTCHA answer). |
| No mouse = penalised? | Without a usable *mouse* trajectory the model is **not applied**; the session starts from a neutral prior (MEDIUM). Never HIGH on that alone. |
| Weak medium tier | Tiers are concrete (see diagram). HIGH adds a warped 7-char CAPTCHA, PoW, min-wait, lockout with exponential back-off. |
| Continuous evaluation | Each login recomputes risk; a session that behaved and then floods is sent back to re-assess; escalation never relaxes in-session. |
| Probing the classifier | Re-assessment penalty, lockout blocks `/api/assess`, hard per-IP 429 limit. |
| Replay attacks | Exact-duplicate trajectory hash + held-out replay evaluation. Jittered replays are only partly caught, see limitations. |
| Privacy | No key identities/text recorded; only geometry/timing/counts; collection logging is opt-in (`CAPTCHA_COLLECT=1`). Add consent text + retention policy for a real deployment (DPDP/GDPR). |

## Known limitations (be upfront about these in the report)
* **Verified in the sandbox:** feature code, loaders on *fake* files in the documented formats, training pipeline, 10 app tests,
  live HTTP run with scripted bots, JS SHA-256 and PoW cross-checked against Python.
  **Not verified:** the DELBOT trajectory text layout (undocumented; auto-detected) and the UI in a real browser.
* Numbers from `--synthetic` mean nothing; synthetic bots are easy to separate.
* **Replay with jitter defeats behavioural-only detection** by construction; a determined attacker with real
  human traces will pass LOW. Defences are server-side (rate, PoW cost, duplicate/near-duplicate detection),
  not the classifier.
* Training data is mouse-only on a different task from a login form; there is no touch/keyboard model.
  Keyboard-only and touch users always get at least a MEDIUM CAPTCHA (a usability cost to report).
* Thresholds reflect dataset class balance, not real traffic priors. Re-tune on live data.
* In-memory state and rate limits are per-process: use Redis and a proxy-aware client IP in production.
* PoW difficulty (12/18 bits) is a demo setting, not a tuned economic cost.

## Layout
`app.py` · `risk/{features,model,engine,challenges}.py` · `static/collector.js` · `templates/index.html` ·
`training/{loaders,synth,train,inspect_data}.py` · `tools/{bots,summarize_collected}.py` · `tests/`

## Citations
CaptchaSolve30k (Capycap Inc., 2026): the author asks to be emailed if it is used in a paper and provides a
citation entry on the dataset card. DELBOT-Mouse: github.com/chrisgdt/DELBOT-Mouse (Apache-2.0).

## Datasets (not included in this repository)
- DELBOT-Mouse: https://github.com/chrisgdt/DELBOT-Mouse (Apache 2.0)
- CaptchaSolve30k: https://huggingface.co/datasets/Capycap-AI/CaptchaSolve30k (Apache 2.0)

Download them into `data/` using the commands in the Quick start section above.
