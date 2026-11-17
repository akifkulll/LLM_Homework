"""Transformer (technique area C): fine-tune BERTurk (dbmdz/bert-base-turkish-cased) with a
classification head on OffensEval-TR. Run on a GPU (Colab A100: ~5 minutes for 3 epochs).

Run:  python -m src.train_bert [--epochs 3] [--lr 2e-5] [--batch 32] [--quick]
"""
import argparse
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

from .config import BERT_BASE, BERT_HF_DIR, BERT_MAX_LEN, SEED
from .data import load_splits
from .normalize import for_bert
from .utils import best_threshold, macro_f1, save_threshold, seed_everything


def batches(tok, texts, labels, bs, shuffle):
    data = list(zip([for_bert(t) for t in texts], labels))

    def collate(b):
        enc = tok([x for x, _ in b], truncation=True, max_length=BERT_MAX_LEN, padding=True, return_tensors="pt")
        enc["labels"] = torch.tensor([y for _, y in b])
        return enc

    g = torch.Generator().manual_seed(SEED)
    return DataLoader(data, batch_size=bs, shuffle=shuffle, collate_fn=collate, generator=g)


@torch.no_grad()
def predict(model, loader, device, amp_dtype):
    model.eval()
    probs = []
    for b in loader:
        b = {k: v.to(device) for k, v in b.items() if k != "labels"}
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(**b).logits
        probs.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--quick", action="store_true", help="smoke test on 1000 examples")
    args = ap.parse_args()

    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp_dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else None
    print(f"device={device}  amp={amp_dtype}")

    s = load_splits()
    if args.quick:
        s = {k: (x[:1000], y[:1000]) for k, (x, y) in s.items()}
        args.epochs = 1

    tok = AutoTokenizer.from_pretrained(BERT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE, num_labels=2, id2label={0: "NOT", 1: "OFF"}, label2id={"NOT": 0, "OFF": 1}).to(device)

    train_dl = batches(tok, *s["train"], args.batch, shuffle=True)
    val_dl = batches(tok, *s["val"], 128, shuffle=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total = len(train_dl) * args.epochs
    sched = get_linear_schedule_with_warmup(opt, int(0.06 * total), total)

    best_f = -1.0
    for ep in range(args.epochs):
        model.train()
        t0, run = time.time(), 0.0
        for step, b in enumerate(train_dl, 1):
            b = {k: v.to(device) for k, v in b.items()}
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=amp_dtype is not None):
                loss = model(**b).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            run += loss.item()
            if step % 200 == 0:
                print(f"  epoch {ep + 1} step {step}/{len(train_dl)}  loss={run / step:.4f}")
        pva = predict(model, val_dl, device, amp_dtype)
        t, f = best_threshold(s["val"][1], pva)
        print(f"epoch {ep + 1}: train loss={run / len(train_dl):.4f}  val macro-F1={f:.4f} "
              f"(threshold {t})  [{time.time() - t0:.0f}s]")
        if f > best_f:
            best_f = f
            BERT_HF_DIR.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(BERT_HF_DIR)
            tok.save_pretrained(BERT_HF_DIR)
            save_threshold("bert", t)
            print(f"  saved best checkpoint -> {BERT_HF_DIR}")

    # reference score of the best PyTorch checkpoint on the test set
    model = AutoModelForSequenceClassification.from_pretrained(BERT_HF_DIR).to(device)
    from .utils import load_threshold
    t = load_threshold("bert")
    pte = predict(model, batches(tok, *s["test"], 128, shuffle=False), device, amp_dtype)
    print(f"best val macro-F1={best_f:.4f}; test macro-F1 (PyTorch, reference)="
          f"{macro_f1(s['test'][1], (pte >= t).astype(int)):.4f}")


if __name__ == "__main__":
    main()
