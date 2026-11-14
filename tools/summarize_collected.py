"""Usability/security summary of sessions logged with CAPTCHA_COLLECT=1.
    python -m tools.summarize_collected collected/sessions.jsonl
"""
import json
import sys

import pandas as pd

rows = [json.loads(l) for l in open(sys.argv[1] if len(sys.argv) > 1 else "collected/sessions.jsonl")]
df = pd.DataFrame(rows)
df["who"] = df["label"].where(df["label"] == "human", df["bot_type"])
t = pd.crosstab(df["who"], df["tier"], normalize="index").reindex(columns=["low", "medium", "high"], fill_value=0).round(3)
t["n"] = df.groupby("who").size()
print(t)
h = df[df.label == "human"]
if len(h):
    print(f"\nUSABILITY: {(h.tier != 'low').mean():.1%} of human sessions saw a visible CAPTCHA or worse "
          f"({(h.tier == 'high').mean():.1%} got the hardest tier).")
print("By pointer mode (humans):\n", h.groupby("pointer_type").tier.value_counts(normalize=True).round(3) if len(h) else "-")
