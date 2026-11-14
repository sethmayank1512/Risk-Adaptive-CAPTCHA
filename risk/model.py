"""Loads the trained trajectory model (or a clearly-labelled heuristic fallback)."""
import json
import math
from pathlib import Path

import numpy as np

DEFAULT_LOW, DEFAULT_HIGH = 0.30, 0.70


def _sigmoid(z):
    return 1.0 / (1.0 + math.exp(-z))


class TrajModel:
    def __init__(self, art_dir="artifacts"):
        self.model, self.features, self.meta, self.fallback = None, None, {}, True
        p, m = Path(art_dir) / "traj_model.joblib", Path(art_dir) / "model_meta.json"
        if p.exists() and m.exists():
            import joblib
            blob = joblib.load(p)
            self.model, self.features = blob["model"], blob["features"]
            self.meta = json.loads(m.read_text())
            self.fallback = False
        self.low = float(self.meta.get("low_threshold", DEFAULT_LOW))
        self.high = float(self.meta.get("high_threshold", DEFAULT_HIGH))
        self.synthetic = bool(self.meta.get("synthetic", False))

    def banner(self):
        if self.fallback:
            return "No trained model found - using a crude heuristic. Run `python -m training.train`."
        if self.synthetic:
            return "DEMO MODEL: trained on synthetic data only. Retrain on real datasets before drawing conclusions."
        return ""

    def score(self, f):
        """Probability that the trajectory belongs to a bot."""
        if self.fallback:
            z = (-1.5 + 3.0 * (f["straightness"] - 0.7) + 6.0 * (f["turn_zero_frac"] - 0.1)
                 + 2.0 * (0.5 - min(f["v_cv"], 1.0)))
            return _sigmoid(z)
        x = np.array([[f[k] for k in self.features]], float)
        return float(self.model.predict_proba(np.nan_to_num(x))[0, 1])
