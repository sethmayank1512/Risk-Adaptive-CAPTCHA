"""Trade-off plot: how many humans get bothered vs how many bots get through, as the threshold moves."""
import numpy as np
import matplotlib.pyplot as plt


def _curve(hum, bot):
    ts = np.append(np.linspace(0, 1, 401), 1.0001)
    hs, bs = hum.score.to_numpy(), bot.score.to_numpy()
    return np.array([(hs >= t).mean() for t in ts]), np.array([(bs < t).mean() for t in ts])


def run(te, lo, hi, rep, lines, seed=0):
    bots = te[te.label == 1]
    known, replay = bots[bots.bot_type != "syn_replay"], bots[bots.bot_type == "syn_replay"]
    hum_d = te[(te.label == 0) & (te.source == "delbot")]
    hum_c = te[(te.label == 0) & (te.source == "capycap")]
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    for hum, bot, lab, kw in ((hum_d, known, "DELBOT humans vs known bots", dict(c="tab:blue")),
                              (hum_c, known, "CaptchaSolve humans vs known bots", dict(c="tab:orange")),
                              (hum_d, replay, "DELBOT humans vs replay bots", dict(c="tab:red", ls="--"))):
        if len(hum) and len(bot):
            x, y = _curve(hum, bot)
            ax.plot(x, y, label=lab, **kw)
    if len(hum_d) and len(known):
        for t, m, nm in ((lo, "o", "Low/Medium cut"), (hi, "s", "Medium/High cut")):
            ax.plot((hum_d.score >= t).mean(), (known.score < t).mean(), m, c="k", ms=8, label=f"this system: {nm}")
    ax.set(xlabel="Humans bothered (share seeing a CAPTCHA)", ylabel="Bots getting through",
           title="Usability vs security (lower-left is better)", xlim=(-0.02, 1.02), ylim=(-0.02, 1.02))
    ax.legend(fontsize=7, loc="upper right"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{rep}/tradeoff.png", dpi=140); plt.close(fig)
    lines += ["## Trade-off curve", "Each point on a curve is one threshold: right = more humans bothered, "
              "up = more bots getting through. Black markers are this system's two cut-offs.", "",
              "![tradeoff](tradeoff.png)", ""]
    print("Saved " + str(rep) + "/tradeoff.png")
