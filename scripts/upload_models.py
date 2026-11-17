"""Upload models/berturk_onnx/ to the Hugging Face Hub (run in Colab after src.export_onnx).

Usage:  python scripts/upload_models.py --repo <hf-user>/<repo-name>
Log in first (Colab: `from huggingface_hub import notebook_login; notebook_login()`).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import BERT_ONNX_DIR  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--private", action="store_true")
    a = ap.parse_args()
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(a.repo, private=a.private, exist_ok=True)
    api.upload_folder(folder_path=str(BERT_ONNX_DIR), repo_id=a.repo,
                      commit_message="BERTurk fine-tuned on OffensEval-TR: ONNX FP32 + INT8")
    print(f"Uploaded -> https://huggingface.co/{a.repo}")
