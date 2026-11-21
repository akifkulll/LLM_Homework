# Real-Time Turkish Chat Moderation

SWE015 homework, Use Case 4. A chat interface that labels every message as **clean / offensive**
in real time on a **CPU**, masks offensive words and logs per-message latency.
The app runs a **fine-tuned BERTurk quantized to INT8 (ONNX Runtime)** by default. The other
models of the ladder can be selected and compared side by side.

| Ladder step | Model | Technique area |
|---|---|---|
| Rule baseline | Regex normalization + lexicon matcher with minimum edit distance | B |
| Classical baseline | TF-IDF (word 1-2 grams + char 2-5 grams) + logistic regression | B |
| Neural network | Own word2vec (skip-gram, negative sampling) → averaged vectors → feed-forward net (SGD + backprop) | A |
| Transformer | BERTurk (`dbmdz/bert-base-turkish-cased`) fine-tuned with a classification head | C |
| Quantized Transformer | BERTurk → ONNX → INT8 dynamic quantization (ONNX Runtime) | D |

Data: [OffensEval-TR 2020](https://coltekin.github.io/offensive-turkish/) (Çöltekin, 2020), 31,756 training
tweets and 3,528 test tweets. All models are evaluated on the official test set. The validation split is
10% of the training file (deterministic, by tweet id hash). Metric: macro-F1 (classes are imbalanced, ~20% offensive).

## Setup

```bash
conda create -n chatmod python=3.12 -y
conda activate chatmod
pip install -r requirements.txt
```

## Run the app (single command)

```bash
python run.py
```

`run.py` downloads the dataset, trains the small CPU models if they are missing (~3 min),
downloads the fine-tuned BERTurk ONNX files (FP32 + INT8, ~530 MB) from the Hugging Face Hub
([akifkulll/berturk-offenseval-tr-onnx](https://huggingface.co/akifkulll/berturk-offenseval-tr-onnx)),
warms up all models and opens the Gradio app at http://127.0.0.1:7860.
The fine-tuned PyTorch checkpoint is at [akifkulll/berturk-offenseval-tr](https://huggingface.co/akifkulll/berturk-offenseval-tr).

Results table (macro-F1, size, latency) on your own CPU: `python -m src.benchmark`
(the numbers in the presentation are in [`results/results.md`](results/results.md)).

## Reproduce everything

| Step | Where | Command |
|---|---|---|
| Data stats | any | `python -m src.data` |
| TF-IDF baseline | CPU | `python -m src.train_tfidf` |
| word2vec + FFN | CPU | `python -m src.train_w2v_ffn` |
| BERTurk fine-tune | GPU (Colab) | `python -m src.train_bert` |
| ONNX export + INT8 quantization | Colab | `python -m src.export_onnx` |
| Upload / download ONNX models | Colab / local | `scripts/upload_models.py`, `scripts/download_models.py` |
| Results table (F1, size, latency) | target CPU | `python -m src.benchmark` |

The Colab steps are in [`notebooks/train_berturk_colab.ipynb`](notebooks/train_berturk_colab.ipynb).
`src.benchmark` writes `results/results.md`, `results/results.json` and the misclassified test messages
of every model (`results/errors_*.csv`).

## Project layout

```
app.py                 Gradio chat UI (default model: BERTurk INT8)
run.py                 single run command
src/normalize.py       regex normalization (leetspeak, repeated letters, a.m.k, Turkish chars)
src/lexicon.py         lexicon matcher + minimum edit distance (rule baseline)
src/train_tfidf.py     TF-IDF + logistic regression
src/train_w2v_ffn.py   own word2vec + feed-forward classifier
src/train_bert.py      BERTurk fine-tuning
src/export_onnx.py     ONNX export + INT8 quantization + sanity checks
src/models.py          common inference interface for all models
src/benchmark.py       evaluation on the test set: macro-F1, size, latency
data/lexicon_tr.txt    hand-collected offensive-word lexicon
```

## Sources

- Çağrı Çöltekin. 2020. *A Corpus of Turkish Offensive Language on Social Media.* LREC 2020. (OffensEval-TR 2020 data)
- Stefan Schweter. *BERTurk: BERT models for Turkish.* `dbmdz/bert-base-turkish-cased` on the Hugging Face Hub.
