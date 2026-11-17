"""Single run command: prepares whatever is missing, then starts the app.

  1. downloads OffensEval-TR (if needed)
  2. trains the tf-idf baseline and word2vec + FFN on CPU (if needed, ~3 minutes)
  3. downloads the BERTurk ONNX models from the Hugging Face Hub (if needed and HF_REPO is set)
  4. launches the Gradio app

Usage:  python run.py [--repo <hf-user>/<repo-name>]
"""
import argparse

from src.config import BERT_INT8_ONNX, HF_REPO, TFIDF_PATH, W2V_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=HF_REPO, help="Hugging Face repo with the BERTurk ONNX files")
    a = ap.parse_args()

    from src import data
    data.download()
    if not TFIDF_PATH.exists():
        print("== training tf-idf baseline ==")
        from src import train_tfidf
        train_tfidf.main()
    if not (W2V_DIR / "ffn.pt").exists():
        print("== training word2vec + FFN ==")
        from src import train_w2v_ffn
        train_w2v_ffn.main()
    if not BERT_INT8_ONNX.exists():
        if a.repo:
            from scripts.download_models import download
            download(a.repo)
        else:
            print("!! BERTurk ONNX files not found and no --repo given: the app starts without BERT models.")

    import app
    app.launch()


if __name__ == "__main__":
    main()
