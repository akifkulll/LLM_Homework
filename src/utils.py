"""Shared helpers: metrics, threshold selection, seeding, threshold store."""
import json
import random

import numpy as np
from sklearn.metrics import f1_score

from .config import THRESHOLDS_PATH


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro"))


def best_threshold(y_true, probs):
    """Decision threshold on P(offensive) that maximizes macro-F1 on the VALIDATION set.
    Classes are imbalanced (~20% offensive), so 0.5 is rarely the best cut-off."""
    probs = np.asarray(probs)
    best_t, best_f = 0.5, -1.0
    for t in np.arange(0.05, 0.951, 0.01):
        f = macro_f1(y_true, (probs >= t).astype(int))
        if f > best_f:
            best_t, best_f = float(round(t, 2)), f
    return best_t, best_f


def save_threshold(key: str, value: float):
    THRESHOLDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(THRESHOLDS_PATH.read_text()) if THRESHOLDS_PATH.exists() else {}
    data[key] = value
    THRESHOLDS_PATH.write_text(json.dumps(data, indent=2))


def load_threshold(key: str, default: float = 0.5) -> float:
    if THRESHOLDS_PATH.exists():
        return float(json.loads(THRESHOLDS_PATH.read_text()).get(key, default))
    return default
