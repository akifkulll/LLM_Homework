"""Download and split OffensEval-TR 2020 (Coltekin, 2020).

Splits
  train / val : the official training file, split 90/10 by a hash of the tweet id
                (deterministic on every machine and library version)
  test        : the official OffensEval-TR 2020 test set with gold labels
Every model in this project is evaluated on exactly this test set.
"""
import hashlib
import io
import zipfile

from .config import RAW_DIR

URL = "https://coltekin.github.io/offensive-turkish/offenseval2020-turkish.zip"
LABEL2ID = {"NOT": 0, "OFF": 1}


def download():
    if any(RAW_DIR.rglob("offenseval-tr-training-v1.tsv")):
        return
    from urllib.request import urlopen

    print(f"Downloading {URL} ...")
    with urlopen(URL, timeout=300) as r:
        payload = r.read()
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(io.BytesIO(payload)).extractall(RAW_DIR)


def _find(name):
    hits = list(RAW_DIR.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {RAW_DIR}")
    return hits[0]


def _lines(path):
    with open(path, encoding="utf-8") as f:
        return [ln.rstrip("\r\n") for ln in f if ln.strip()]


def _read_train():
    rows = []
    for ln in _lines(_find("offenseval-tr-training-v1.tsv")):
        parts = ln.split("\t")
        if parts[0] == "id":
            continue
        tid, label, text = parts[0], parts[-1].strip(), "\t".join(parts[1:-1])
        rows.append((tid, text, LABEL2ID[label]))
    return rows


def _read_test():
    texts = []
    for ln in _lines(_find("offenseval-tr-testset-v1.tsv")):
        parts = ln.split("\t")
        if parts[0] == "id":
            continue
        texts.append((parts[0], "\t".join(parts[1:])))
    labels = {}
    order = []
    for ln in _lines(_find("offenseval-tr-labela-v1.tsv")):
        parts = [p.strip() for p in ln.replace("\t", ",").split(",")]
        if parts[-1] not in LABEL2ID:
            continue  # header
        labels[parts[0]] = LABEL2ID[parts[-1]]
        order.append(LABEL2ID[parts[-1]])
    if all(tid in labels for tid, _ in texts):
        return [(tid, t, labels[tid]) for tid, t in texts]
    assert len(order) == len(texts), "test labels do not line up with test tweets"
    return [(tid, t, y) for (tid, t), y in zip(texts, order)]


def _is_val(tid):
    return int(hashlib.md5(tid.encode()).hexdigest(), 16) % 10 == 0


def load_splits():
    """Returns {"train"|"val"|"test": (texts, labels)}."""
    download()
    train_rows = _read_train()
    test_rows = _read_test()
    tr = [r for r in train_rows if not _is_val(r[0])]
    va = [r for r in train_rows if _is_val(r[0])]
    out = {}
    for name, rows in (("train", tr), ("val", va), ("test", test_rows)):
        out[name] = ([r[1] for r in rows], [r[2] for r in rows])
    return out


def describe(splits):
    for k, (x, y) in splits.items():
        off = sum(y)
        print(f"{k:5s}: {len(x):6d} messages | offensive {off:5d} ({off / len(y):.1%})")


if __name__ == "__main__":
    describe(load_splits())
