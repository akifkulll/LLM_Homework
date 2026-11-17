"""Classical baseline: tf-idf (word 1-2 grams + character 2-5 grams) + logistic regression.

Run:  python -m src.train_tfidf
"""
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline

from .config import MODELS_DIR, SEED, TFIDF_PATH
from .data import load_splits
from .normalize import for_classic
from .utils import best_threshold, macro_f1, save_threshold, seed_everything


def build(C: float) -> Pipeline:
    features = FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, token_pattern=r"\S+")),
        # character n-grams are robust to Turkish suffixes and misspellings
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True,
                                 max_features=200_000)),
    ])
    clf = LogisticRegression(C=C, class_weight="balanced", max_iter=3000, random_state=SEED)
    return Pipeline([("tfidf", features), ("lr", clf)])


def main():
    seed_everything(SEED)
    s = load_splits()
    xtr = [for_classic(t) for t in s["train"][0]]
    xva = [for_classic(t) for t in s["val"][0]]
    ytr, yva = s["train"][1], s["val"][1]

    best = None
    for C in (0.5, 1.0, 2.0, 4.0, 8.0):
        model = build(C).fit(xtr, ytr)
        t, f = best_threshold(yva, model.predict_proba(xva)[:, 1])
        print(f"C={C:<4}  val macro-F1={f:.4f}  (threshold {t})")
        if best is None or f > best[0]:
            best = (f, C, t, model)

    f, C, t, model = best
    print(f"-> selected C={C}, threshold={t}, val macro-F1={f:.4f}")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, TFIDF_PATH, compress=3)
    save_threshold("tfidf", t)

    xte = [for_classic(x) for x in s["test"][0]]
    p = (model.predict_proba(xte)[:, 1] >= t).astype(int)
    print(f"test macro-F1 (reference; final numbers come from src.benchmark): {macro_f1(s['test'][1], p):.4f}")


if __name__ == "__main__":
    main()
