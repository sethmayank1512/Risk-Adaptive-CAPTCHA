"""Train + evaluate the trajectory risk model.

Examples
    # real data
    python -m training.train --delbot data/DELBOT-Mouse --capycap data/CaptchaSolve30k/data.jsonl
    # pipeline smoke test with fake data (model is tagged synthetic / demo-only)
    python -m training.train --synthetic

What it does (see reports/report.md afterwards)
  1. load sessions, split by *group* (never by window) into train/val/test, per source+bot-type stratum
  2. crop random 1.5-6 s windows (login-page-like) and extract rate-invariant features
  3. add synthetic bot tiers (naive/bezier/noisy-bezier); replay bots are TEST-ONLY
  4. compare LogisticRegression / RandomForest / HistGradientBoosting
  5. source-leakage check: can features tell DELBOT humans from CaptchaSolve humans?
  6. leave-one-bot-type-out: how well are unseen bot families detected?
  7. final model: calibrate on val, choose Low/High thresholds from held-out HUMAN scores,
     report tier rates for each human source and each bot type on the untouched test split
"""
import argparse
import datetime as dt
import json
import tempfile
from collections import defaultdict
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from risk.features import INVARIANT_FEATURES, extract
from training import synth
from training.loaders import Session, load_capycap, load_collected, load_delbot

FEATS = INVARIANT_FEATURES


# ----------------------------------------------------------------- data prep
def split_sessions(sessions, rng, frac=(0.6, 0.2, 0.2)):
    strata = defaultdict(list)
    for s in sessions:
        strata[(s.source, s.bot_type)].append(s)
    out = {"train": [], "val": [], "test": []}
    for lst in strata.values():
        groups = sorted({s.group for s in lst})
        rng.shuffle(groups)
        n = len(groups)
        if n < 3:
            m = {g: "train" for g in groups}
        else:
            a = max(1, int(frac[0] * n)); b = max(a + 1, int((frac[0] + frac[1]) * n))
            m = {g: ("train" if i < a else "val" if i < b else "test") for i, g in enumerate(groups)}
        for s in lst:
            out[m[s.group]].append(s)
    return out


def to_frame(sessions, rng, k=3, win=(1.5, 6.0)):
    rows = []
    for s in sessions:
        tr = s.traj
        t0, t1 = tr[0, 2], tr[-1, 2]
        for w in range(k):
            L = rng.uniform(*win) * 1000
            if t1 - t0 <= L:
                seg = tr
            else:
                st = rng.uniform(t0, t1 - L)
                seg = tr[(tr[:, 2] >= st) & (tr[:, 2] <= st + L)]
            f = extract(seg)
            if f:
                rows.append({**f, "label": s.label, "source": s.source, "bot_type": s.bot_type, "group": s.group})
            if t1 - t0 <= L:
                break
    return pd.DataFrame(rows)


def synth_sessions(rng, humans, tag, kinds, n_each, replay=False):
    out = []
    for kd in kinds:
        for i in range(n_each):
            out.append(Session(synth.synth_bot(rng, kd), 1, "synthetic", f"syn_{kd}", f"syn:{tag}:{kd}:{i}"))
    if replay and humans:
        idx = rng.choice(len(humans), size=min(n_each * 2, len(humans)), replace=False)
        for j, i in enumerate(idx):
            out.append(Session(synth.synth_replay(rng, humans[i].traj), 1, "synthetic", "syn_replay", f"syn:{tag}:replay:{j}"))
    return out


# ----------------------------------------------------------------- models
def make_models(seed):
    return {
        "logreg": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", max_iter=3000)),
        "random_forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, max_features="sqrt",
                                                class_weight="balanced_subsample", n_jobs=-1, random_state=seed),
        "hist_gb": HistGradientBoostingClassifier(max_depth=4, learning_rate=0.1, class_weight="balanced", random_state=seed),
    }


def XY(df):
    return np.nan_to_num(df[FEATS].to_numpy(float), nan=0.0, posinf=1e3, neginf=-1e3), df["label"].to_numpy()


def tpr_at_fpr(y, s, fpr_max):
    fpr, tpr, _ = roc_curve(y, s)
    return float(tpr[fpr <= fpr_max].max()) if (fpr <= fpr_max).any() else 0.0


def metrics(y, s):
    return {"auc": float(roc_auc_score(y, s)), "ap": float(average_precision_score(y, s)),
            "tpr@fpr1%": tpr_at_fpr(y, s, 0.01), "tpr@fpr5%": tpr_at_fpr(y, s, 0.05), "n": int(len(y))}


def thresholds(human_scores, low_q, high_q):
    low = float(np.clip(np.quantile(human_scores, low_q), 0.02, 0.30))
    high = float(np.clip(np.quantile(human_scores, high_q), 0.50, 0.95))
    return low, max(high, low + 0.15)


def tier(scores, low, high):
    return np.where(scores < low, "low", np.where(scores >= high, "high", "medium"))


def tier_table(df, scores, low, high, by):
    t = pd.DataFrame({"g": df[by].values, "tier": tier(scores, low, high)})
    tab = pd.crosstab(t["g"], t["tier"], normalize="index").reindex(columns=["low", "medium", "high"], fill_value=0)
    tab["n"] = t.groupby("g").size()
    return tab.round(3)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delbot"); ap.add_argument("--capycap"); ap.add_argument("--collected")
    ap.add_argument("--synthetic", action="store_true", help="use fake data (pipeline test only)")
    ap.add_argument("--final", choices=["delbot", "mixed"], default="delbot",
                    help="'mixed' also trains on CaptchaSolve humans (see report for the leakage check)")
    ap.add_argument("--model", choices=["random_forest", "logreg", "hist_gb"], default="random_forest")
    ap.add_argument("--max-capycap", type=int, default=3000)
    ap.add_argument("--screen", type=int, nargs=2, default=[1920, 1080])
    ap.add_argument("--delbot-layout", choices=["auto", "txy", "xyt"], default="auto",
                    help="column order of DELBOT text files if auto-detection is wrong")
    ap.add_argument("--capycap-canvas", type=float, nargs=2)
    ap.add_argument("--capycap-field"); ap.add_argument("--capycap-fmt"); ap.add_argument("--capycap-stride", type=int)
    ap.add_argument("--capycap-source", choices=["ticks", "stream"], default="ticks",
                    help="'ticks' = documented tickInputs JSON (default); 'stream' = binary inputStream fallback")
    ap.add_argument("--capycap-include-touch", action="store_true", help="also use touchscreen sessions (model is mouse-only: not recommended)")
    ap.add_argument("--no-augment", action="store_true", help="do not add synthetic bot tiers")
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--human-low-q", type=float, default=0.90, help="share of held-out humans that should land in LOW")
    ap.add_argument("--human-high-q", type=float, default=0.99, help="max share of held-out humans allowed in HIGH = 1-this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="artifacts"); ap.add_argument("--report", default="reports")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out, rep = Path(a.out), Path(a.report)
    out.mkdir(exist_ok=True); rep.mkdir(exist_ok=True)

    # 1. load -------------------------------------------------------------
    tmp = None
    if a.synthetic:
        tmp = tempfile.mkdtemp()
        a.delbot = synth.write_fake_delbot_tree(tmp, n=400, seed=a.seed)
        synth.write_fake_capycap(f"{tmp}/c.jsonl", n=300, seed=a.seed + 1)
        a.capycap = f"{tmp}/c.jsonl"
        print("!! SYNTHETIC MODE: results below only test the pipeline, they are not research results.")
    if not a.delbot:
        ap.error("--delbot is required (or use --synthetic)")
    delbot = load_delbot(a.delbot, screen=tuple(a.screen), layout=a.delbot_layout)
    capy = load_capycap(a.capycap, a.max_capycap, tuple(a.capycap_canvas) if a.capycap_canvas else None,
                        field=a.capycap_field, fmt=a.capycap_fmt, stride=a.capycap_stride,
                        include_touch=a.capycap_include_touch, source=a.capycap_source) if a.capycap else []
    print(f"DELBOT: {sum(s.label==0 for s in delbot)} human / {sum(s.label==1 for s in delbot)} bot sessions; "
          f"CaptchaSolve: {len(capy)} human sessions")
    if not any(s.label == 1 for s in delbot) or not any(s.label == 0 for s in delbot):
        raise SystemExit("DELBOT loader did not yield both classes - check labels with training.inspect_data")

    # 2. split by group, then synthesise augmentation per split ---------------
    sp = split_sessions(delbot + capy, rng)
    kinds = [] if a.no_augment else ["naive", "naive_dense", "bezier", "noisy_bezier"]
    for name in sp:
        hum = [s for s in sp[name] if s.label == 0]
        d_h = [s for s in hum if s.source == "delbot"]
        n_each = max(30, int(0.3 * len(d_h)))
        if kinds:
            sp[name] += synth_sessions(rng, d_h, name, kinds, n_each, replay=(name == "test"))
    frames = {k: to_frame(v, rng, a.windows) for k, v in sp.items()}
    for k, df in frames.items():
        print(f"{k}: {len(df)} windows | " + ", ".join(f"{b}={n}" for b, n in df.groupby('bot_type').size().items()))
    tr, va, te = frames["train"], frames["val"], frames["test"]
    REPLAY = "syn_replay"

    def trainable(df, include_capy):
        m = (df["bot_type"] != REPLAY)
        if not include_capy:
            m &= (df["source"] != "capycap")
        return df[m]

    # label sanity: bots should look different from humans on a couple of robust features
    san = tr[tr.source != "capycap"].groupby("label")[["straightness", "turn_zero_frac", "micro_jitter"]].mean().round(3)
    print("Label sanity (0=human, 1=bot) - verify direction makes sense:\n", san)

    results = {"synthetic": bool(a.synthetic), "args": vars(a) | {"delbot": str(a.delbot), "capycap": str(a.capycap)}}
    lines = ["# Risk model report", f"_generated {dt.datetime.now():%Y-%m-%d %H:%M}_",
             "> **SYNTHETIC RUN - NOT RESULTS**" if a.synthetic else "", "",
             "## Data", "```", *[f"{k}: {len(df)} windows" for k, df in frames.items()], "```", ""]

    # 3. model comparison on in-domain test -------------------------------
    Xtr, ytr = XY(trainable(tr, False))
    in_dom = te[(te["source"] != "capycap") & (te["bot_type"] != REPLAY)]
    del_only = te[te["source"] == "delbot"]
    cmp_rows = {}
    roc_fig, roc_ax = plt.subplots(1, 2, figsize=(10, 4))
    for name, mdl in make_models(a.seed).items():
        mdl.fit(Xtr, ytr)
        row = {}
        for tag, d in (("delbot_only", del_only), ("in_domain_all_bots", in_dom)):
            X, y = XY(d)
            s = mdl.predict_proba(X)[:, 1]
            row[tag] = metrics(y, s)
            if tag == "delbot_only":
                fpr, tpr, _ = roc_curve(y, s); roc_ax[0].plot(fpr, tpr, label=f"{name} (AUC {row[tag]['auc']:.3f})")
                p, r, _ = precision_recall_curve(y, s); roc_ax[1].plot(r, p, label=name)
        cmp_rows[name] = row
    roc_ax[0].set(xlabel="FPR (humans flagged)", ylabel="TPR (bots caught)", title="ROC - DELBOT test"); roc_ax[0].legend()
    roc_ax[1].set(xlabel="recall", ylabel="precision", title="PR - DELBOT test"); roc_ax[1].legend()
    roc_fig.tight_layout(); roc_fig.savefig(rep / "roc_pr.png", dpi=130); plt.close(roc_fig)
    results["model_comparison"] = cmp_rows
    flat = pd.DataFrame({(t, m): v for n, r in cmp_rows.items() for t, v in r.items() for m in [n]}).T
    lines += ["## Model comparison (test split, group-disjoint)", "```", pd.DataFrame(
        [{"model": n, "set": t, **v} for n, r in cmp_rows.items() for t, v in r.items()]).round(4).to_string(index=False), "```", ""]

    # 4. source leakage check ------------------------------------------------
    if capy:
        allh = pd.concat([f[(f.label == 0) & (f.source.isin(["delbot", "capycap"]))] for f in frames.values()])
        yl = (allh["source"] == "capycap").astype(int).to_numpy()
        Xl = np.nan_to_num(allh[FEATS].to_numpy(float))
        tri, tei = next(GroupShuffleSplit(1, test_size=0.3, random_state=a.seed).split(Xl, yl, allh["group"]))
        rf = make_models(a.seed)["random_forest"].fit(Xl[tri], yl[tri])
        auc = roc_auc_score(yl[tei], rf.predict_proba(Xl[tei])[:, 1])
        top = pd.Series(rf.feature_importances_, FEATS).sort_values(ascending=False).head(6).round(3)
        results["source_leakage_auc"] = float(auc)
        verdict = ("HIGH - features strongly encode which dataset a human came from; mixing sources as human/bot "
                   "classes would be confounded. Keep DELBOT as the only labelled-class source." if auc > 0.85 else
                   "moderate" if auc > 0.7 else "low - sources are hard to tell apart from these features")
        print(f"Source leakage AUC (DELBOT humans vs CaptchaSolve humans): {auc:.3f} -> {verdict}")
        lines += ["## Source-leakage check (humans only: DELBOT vs CaptchaSolve30k)", f"AUC = **{auc:.3f}** - {verdict}",
                  "Most source-revealing features:", "```", top.to_string(), "```", ""]

    # 5. leave-one-bot-type-out ------------------------------------------------
    types = sorted(t for t in tr.bot_type.unique() if t != "human" and t != REPLAY)
    loo = []
    base_tr = trainable(tr, False)
    for T in types:
        sub = base_tr[base_tr.bot_type != T]
        mdl = make_models(a.seed)["random_forest"].fit(*XY(sub))
        vh = va[(va.label == 0) & (va.source == "delbot")]
        lo, hi = thresholds(mdl.predict_proba(XY(vh)[0])[:, 1], a.human_low_q, a.human_high_q)
        tb = te[te.bot_type == T]
        if len(tb) == 0:
            continue
        s = mdl.predict_proba(XY(tb)[0])[:, 1]
        loo.append({"held_out_bot_type": T, "n": len(tb), "not_low": float((s >= lo).mean()), "high": float((s >= hi).mean())})
    if loo:
        results["leave_one_bot_type_out"] = loo
        lines += ["## Leave-one-bot-type-out (bot family never seen in training)",
                  "`not_low` = share that would get at least a CAPTCHA; `high` = share that would get the strongest tier.",
                  "```", pd.DataFrame(loo).round(3).to_string(index=False), "```", ""]

    # 6. final model --------------------------------------------------------
    inc = a.final == "mixed"
    ftr = trainable(tr, inc); fva = trainable(va, inc)
    base = make_models(a.seed)[a.model].fit(*XY(ftr))
    cal = CalibratedClassifierCV(FrozenEstimator(base), method="sigmoid").fit(*XY(fva))
    hv = fva[fva.label == 0]
    lo, hi = thresholds(cal.predict_proba(XY(hv)[0])[:, 1], a.human_low_q, a.human_high_q)
    print(f"Final model={a.model} ({a.final}); thresholds low={lo:.3f} high={hi:.3f}")

    ts = cal.predict_proba(XY(te)[0])[:, 1]
    te = te.assign(score=ts)
    hum_tab = tier_table(te[te.label == 0], te[te.label == 0].score.values, lo, hi, "source")
    bot_tab = tier_table(te[te.label == 1], te[te.label == 1].score.values, lo, hi, "bot_type")
    results["thresholds"] = {"low": lo, "high": hi}
    results["tier_rates_humans"] = hum_tab.reset_index().to_dict("records")
    results["tier_rates_bots"] = bot_tab.reset_index().to_dict("records")
    lines += [f"## Final model: {a.model}, trained on '{a.final}', sigmoid-calibrated",
              f"Thresholds from held-out human scores: LOW < {lo:.3f} <= MEDIUM < {hi:.3f} <= HIGH", "",
              "### Usability cost - where do held-out HUMANS land?", "```", hum_tab.to_string(), "```",
              "`capycap` rows are out-of-distribution humans (different task/device): high medium/high shares there "
              "mean the model would annoy real users whose behaviour differs from DELBOT's collection.", "",
              "### Security - where do held-out BOTS land? (`syn_replay` was never trained on)", "```", bot_tab.to_string(), "```", ""]

    # plots
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.hist(te[te.label == 0].score, bins=40, alpha=.6, label="human", density=True)
    ax.hist(te[te.label == 1].score, bins=40, alpha=.6, label="bot", density=True)
    ax.axvline(lo, c="g", ls="--"); ax.axvline(hi, c="r", ls="--")
    ax.set(xlabel="calibrated bot probability", title="Test scores + tier thresholds"); ax.legend()
    fig.tight_layout(); fig.savefig(rep / "score_hist.png", dpi=130); plt.close(fig)
    imp = getattr(base, "feature_importances_", None)
    if imp is not None:
        s = pd.Series(imp, FEATS).sort_values()
        fig, ax = plt.subplots(figsize=(6, 6)); s.plot.barh(ax=ax); ax.set_title("Feature importance")
        fig.tight_layout(); fig.savefig(rep / "feature_importance.png", dpi=130); plt.close(fig)

    from training.extra_eval import run as extra_eval
    extra_eval(te, lo, hi, rep, lines, a.seed)

    # optional: score the app's own collected sessions
    if a.collected and Path(a.collected).exists():
        cs = load_collected(a.collected)
        cf = to_frame(cs, rng, 1, (6, 6))
        if len(cf):
            cf = cf.assign(score=cal.predict_proba(XY(cf)[0])[:, 1])
            ct = tier_table(cf, cf.score.values, lo, hi, "label")
            lines += ["### App-collected sessions (deployment realism)", "```", ct.to_string(), "```", ""]

    joblib.dump({"model": cal, "features": FEATS}, out / "traj_model.joblib")
    meta = {"low_threshold": lo, "high_threshold": hi, "features": FEATS, "model": a.model, "trained_on": a.final,
            "synthetic": bool(a.synthetic), "created": dt.datetime.now().isoformat(timespec="seconds"),
            "window_s": 6.0, "n_train_windows": int(len(ftr)),
            "note": "Thresholds are operating points on held-out data with dataset class balance; "
                    "re-tune against real traffic before deployment."}
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))
    (rep / "metrics.json").write_text(json.dumps(results, indent=2, default=str))
    (rep / "report.md").write_text("\n".join(lines))
    print(f"\nWrote {out}/traj_model.joblib, {out}/model_meta.json, {rep}/report.md (+ plots)")


if __name__ == "__main__":
    main()
