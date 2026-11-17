"""Inference wrappers with one common interface, used by the benchmark and the app.

Every model exposes:
  name, short             display name / key
  predict_proba(text)     -> P(offensive) for ONE raw message (includes its own preprocessing)
  threshold               decision threshold chosen on the validation set
  files()                 files on disk that make up the model (for the size column)
"""
import json

import numpy as np

from .config import (BERT_FP32_ONNX, BERT_INT8_ONNX, BERT_MAX_LEN, BERT_ONNX_DIR, TFIDF_PATH, W2V_DIR)
from .lexicon import LexiconModel
from .normalize import for_bert, for_classic
from .utils import load_threshold


class TfidfModel:
    name = "TF-IDF + LogReg"
    short = "tfidf"

    def __init__(self):
        import joblib
        self.pipe = joblib.load(TFIDF_PATH)
        self.threshold = load_threshold("tfidf")

    def predict_proba(self, text):
        return float(self.pipe.predict_proba([for_classic(text)])[0, 1])

    def files(self):
        return [TFIDF_PATH]


class W2VFFNModel:
    name = "word2vec + FFN (own)"
    short = "w2v_ffn"

    def __init__(self):
        import torch
        from .train_w2v_ffn import FFN
        torch.set_grad_enabled(False)
        self.torch = torch
        self.emb = np.load(W2V_DIR / "embeddings.npy")
        self.stoi = {w: i for i, w in enumerate(json.loads((W2V_DIR / "vocab.json").read_text(encoding="utf-8")))}
        n = np.load(W2V_DIR / "norm.npz")
        self.mu, self.sd = n["mu"], n["sd"]
        self.net = FFN()
        self.net.load_state_dict(torch.load(W2V_DIR / "ffn.pt", map_location="cpu"))
        self.net.eval()
        self.threshold = load_threshold("w2v_ffn")

    def predict_proba(self, text):
        ids = [self.stoi[w] for w in for_classic(text).split() if w in self.stoi]
        x = self.emb[ids].mean(0) if ids else np.zeros(self.emb.shape[1], dtype=np.float32)
        x = self.torch.tensor(((x - self.mu) / self.sd)[None], dtype=self.torch.float32)
        return float(self.torch.softmax(self.net(x), -1)[0, 1])

    def files(self):
        return [W2V_DIR / f for f in ("embeddings.npy", "vocab.json", "norm.npz", "ffn.pt")]


class OnnxBertModel:
    INPUTS = ("input_ids", "attention_mask", "token_type_ids")

    def __init__(self, path, name, short):
        import onnxruntime as ort
        from transformers import AutoTokenizer
        self.path, self.name, self.short = path, name, short
        self.tok = AutoTokenizer.from_pretrained(BERT_ONNX_DIR)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path.read_bytes(), opts, providers=["CPUExecutionProvider"])
        self.names = {i.name for i in self.sess.get_inputs()}
        thr = BERT_ONNX_DIR / "thresholds.json"
        self.threshold = json.loads(thr.read_text()).get("bert", 0.5) if thr.exists() else load_threshold("bert")

    def predict_proba(self, text):
        enc = self.tok(for_bert(text), truncation=True, max_length=BERT_MAX_LEN, return_tensors="np")
        logits = self.sess.run(None, {k: enc[k].astype(np.int64) for k in self.INPUTS if k in self.names})[0][0]
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())

    def files(self):
        return [self.path]


def load_models(verbose=True):
    """Loads every model whose files exist. Order = the model ladder."""
    makers = [
        ("lexicon", lambda: LexiconModel()),
        ("tfidf", lambda: TfidfModel() if TFIDF_PATH.exists() else None),
        ("w2v_ffn", lambda: W2VFFNModel() if (W2V_DIR / "ffn.pt").exists() else None),
        ("bert_fp32", lambda: OnnxBertModel(BERT_FP32_ONNX, "BERTurk FP32 (ONNX)", "bert_fp32")
         if BERT_FP32_ONNX.exists() else None),
        ("bert_int8", lambda: OnnxBertModel(BERT_INT8_ONNX, "BERTurk INT8 (ONNX, quantized)", "bert_int8")
         if BERT_INT8_ONNX.exists() else None),
    ]
    models = {}
    for key, make in makers:
        m = make()
        if m is None:
            if verbose:
                print(f"[models] {key}: files not found, skipped")
            continue
        models[key] = m
    return models


def size_mb(model) -> float:
    return sum(p.stat().st_size for p in model.files()) / 2 ** 20
