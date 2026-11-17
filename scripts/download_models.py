"""Download the fine-tuned + quantized BERTurk ONNX files from the Hugging Face Hub.

Usage:  python scripts/download_models.py --repo <hf-user>/<repo-name>
(or set HF_REPO in src/config.py)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import BERT_ONNX_DIR, HF_REPO  # noqa: E402


def download(repo: str):
    from huggingface_hub import snapshot_download
    print(f"Downloading BERTurk ONNX files from https://huggingface.co/{repo} -> {BERT_ONNX_DIR}")
    snapshot_download(repo_id=repo, local_dir=str(BERT_ONNX_DIR))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=HF_REPO)
    a = ap.parse_args()
    if not a.repo:
        sys.exit("No repo given: use --repo <user>/<name> or set HF_REPO in src/config.py")
    download(a.repo)
