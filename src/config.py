"""Central paths and settings shared by every script."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
LEXICON_PATH = DATA_DIR / "lexicon_tr.txt"
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"

# Model artifact locations
TFIDF_PATH = MODELS_DIR / "tfidf_lr.joblib"
W2V_DIR = MODELS_DIR / "w2v_ffn"
BERT_HF_DIR = MODELS_DIR / "berturk_fp32_hf"      # PyTorch checkpoint (Colab only)
BERT_ONNX_DIR = MODELS_DIR / "berturk_onnx"       # what the app / benchmark use
BERT_FP32_ONNX = BERT_ONNX_DIR / "model_fp32.onnx"
BERT_INT8_ONNX = BERT_ONNX_DIR / "model_int8.onnx"
THRESHOLDS_PATH = MODELS_DIR / "thresholds.json"

BERT_BASE = "dbmdz/bert-base-turkish-cased"       # BERTurk
BERT_MAX_LEN = 128

# Hugging Face Hub repo that hosts the ONNX BERTurk files (set after uploading).
# Can also be passed as:  python scripts/download_models.py --repo user/name
HF_REPO = "akifkulll/berturk-offenseval-tr-onnx"

SEED = 42
LATENCY_BUDGET_MS = 50.0
LABELS = ["NOT", "OFF"]  # 0 = clean, 1 = offensive
