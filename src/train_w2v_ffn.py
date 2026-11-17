"""Neural model (technique area A): our own word2vec + a feed-forward classifier.

Step 1  word2vec (skip-gram with negative sampling) trained from scratch on the TRAINING messages only.
Step 2  every message -> average of its word vectors -> feed-forward network
        (Linear -> ReLU -> Dropout -> Linear), trained with SGD + backpropagation.

Run:  python -m src.train_w2v_ffn          (CPU is fine, ~2-4 minutes)
"""
import json
import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import SEED, W2V_DIR
from .data import load_splits
from .normalize import for_classic
from .utils import best_threshold, macro_f1, save_threshold, seed_everything

# word2vec hyper-parameters
DIM, WINDOW, NEG, MIN_COUNT, W2V_EPOCHS, SUBSAMPLE = 100, 5, 5, 2, 10, 1e-4
# classifier hyper-parameters
HIDDEN, DROPOUT, LR, MOMENTUM, FFN_EPOCHS, BATCH = 256, 0.3, 0.01, 0.9, 60, 64


# ---------------------------------------------------------------- word2vec
class SkipGram(nn.Module):
    def __init__(self, vocab_size, dim):
        super().__init__()
        self.inp = nn.Embedding(vocab_size, dim, sparse=True)
        self.out = nn.Embedding(vocab_size, dim, sparse=True)
        nn.init.uniform_(self.inp.weight, -0.5 / dim, 0.5 / dim)
        nn.init.zeros_(self.out.weight)

    def forward(self, center, context, negatives):
        v = self.inp(center)                                  # (B, D)
        u = self.out(context)                                 # (B, D)
        n = self.out(negatives)                               # (B, K, D)
        pos = F.logsigmoid((v * u).sum(-1))
        neg = F.logsigmoid(-torch.bmm(n, v.unsqueeze(-1)).squeeze(-1)).sum(-1)
        return -(pos + neg).mean()


def build_vocab(token_lists):
    counts = Counter(t for toks in token_lists for t in toks)
    itos = ["<pad>", "<unk>"] + [w for w, c in counts.most_common() if c >= MIN_COUNT]
    stoi = {w: i for i, w in enumerate(itos)}
    freqs = np.array([0, 0] + [counts[w] for w in itos[2:]], dtype=np.float64)
    return itos, stoi, freqs


def make_pairs(id_lists, freqs, rng):
    total = freqs.sum()
    f = freqs / total
    keep = np.minimum(1.0, np.sqrt(SUBSAMPLE / np.maximum(f, 1e-12)) + SUBSAMPLE / np.maximum(f, 1e-12))
    centers, contexts = [], []
    for ids in id_lists:
        ids = [i for i in ids if i > 1 and rng.random() < keep[i]]   # frequent-word subsampling
        for pos, c in enumerate(ids):
            b = rng.randint(1, WINDOW)                                 # dynamic window
            for j in range(max(0, pos - b), min(len(ids), pos + b + 1)):
                if j != pos:
                    centers.append(c)
                    contexts.append(ids[j])
    return torch.tensor(centers), torch.tensor(contexts)


def train_word2vec(token_lists, device):
    itos, stoi, freqs = build_vocab(token_lists)
    id_lists = [[stoi.get(t, 1) for t in toks] for toks in token_lists]
    noise = torch.tensor(freqs ** 0.75, dtype=torch.float)
    noise = (noise / noise.sum()).to(device)
    model = SkipGram(len(itos), DIM).to(device)
    opt = torch.optim.SparseAdam(model.parameters(), lr=0.003)
    rng = random.Random(SEED)
    print(f"word2vec: vocab={len(itos)}  dim={DIM}  window={WINDOW}  negatives={NEG}")
    for ep in range(W2V_EPOCHS):
        c, x = make_pairs(id_lists, freqs, rng)
        perm = torch.randperm(len(c))
        c, x = c[perm], x[perm]
        bs, total = 2048, 0.0
        steps = math.ceil(len(c) / bs)
        for s in range(steps):
            cb = c[s * bs:(s + 1) * bs].to(device)
            xb = x[s * bs:(s + 1) * bs].to(device)
            nb = torch.multinomial(noise, len(cb) * NEG, replacement=True).view(len(cb), NEG)
            loss = model(cb, xb, nb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
        print(f"  epoch {ep + 1}/{W2V_EPOCHS}  pairs={len(c):,}  loss={total / steps:.4f}")
    emb = model.inp.weight.detach().cpu().numpy()
    emb[0] = 0.0
    return itos, stoi, emb


def nearest(word, stoi, itos, emb, k=6):
    if word not in stoi:
        return []
    e = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    sims = e @ e[stoi[word]]
    return [itos[i] for i in np.argsort(-sims)[1:k + 1]]


# ---------------------------------------------------------------- classifier
class FFN(nn.Module):
    def __init__(self, dim=DIM, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, 2))

    def forward(self, x):
        return self.net(x)


def featurize(texts, stoi, emb):
    """Average of the word vectors of the known tokens (zero vector if none)."""
    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    for i, t in enumerate(texts):
        ids = [stoi[w] for w in for_classic(t).split() if w in stoi]
        if ids:
            out[i] = emb[ids].mean(0)
    return out


def train_ffn(xtr, ytr, xva, yva, device):
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6          # feature standardization (stats from train only)
    norm = lambda a: torch.tensor((a - mu) / sd, dtype=torch.float32)
    Xtr, Xva = norm(xtr).to(device), norm(xva).to(device)
    Ytr = torch.tensor(ytr).to(device)
    w = torch.tensor([1.0, (len(ytr) - sum(ytr)) / sum(ytr)], dtype=torch.float32).to(device)  # class weights
    model = FFN().to(device)
    opt = torch.optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM, weight_decay=1e-4)
    best = (-1, None, 0.5, 0)
    for ep in range(FFN_EPOCHS):
        model.train()
        perm = torch.randperm(len(Xtr), device=device)
        for s in range(0, len(Xtr), BATCH):
            idx = perm[s:s + BATCH]
            loss = F.cross_entropy(model(Xtr[idx]), Ytr[idx], weight=w)
            opt.zero_grad()
            loss.backward()      # backpropagation
            opt.step()           # SGD update
        model.eval()
        with torch.no_grad():
            pva = torch.softmax(model(Xva), -1)[:, 1].cpu().numpy()
        t, f = best_threshold(yva, pva)
        if f > best[0]:
            best = (f, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}, t, ep + 1)
        if (ep + 1) % 5 == 0:
            print(f"  epoch {ep + 1:2d}  loss={loss.item():.4f}  val macro-F1={f:.4f}")
    f, state, t, ep = best
    print(f"-> best epoch {ep}: val macro-F1={f:.4f}, threshold={t}")
    model.load_state_dict(state)
    return model, mu, sd, t


def main():
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    s = load_splits()
    tr_tokens = [for_classic(t).split() for t in s["train"][0]]
    itos, stoi, emb = train_word2vec(tr_tokens, device)
    for w in ("amk", "fenerbahce", "guzel", "salak", "erdogan"):
        print(f"  nearest({w}) = {nearest(w, stoi, itos, emb)}")

    xtr = featurize(s["train"][0], stoi, emb)
    xva = featurize(s["val"][0], stoi, emb)
    model, mu, sd, t = train_ffn(xtr, s["train"][1], xva, s["val"][1], device)

    W2V_DIR.mkdir(parents=True, exist_ok=True)
    np.save(W2V_DIR / "embeddings.npy", emb.astype(np.float32))
    (W2V_DIR / "vocab.json").write_text(json.dumps(itos, ensure_ascii=False), encoding="utf-8")
    np.savez(W2V_DIR / "norm.npz", mu=mu, sd=sd)
    torch.save(model.cpu().state_dict(), W2V_DIR / "ffn.pt")
    save_threshold("w2v_ffn", t)

    model.eval()
    xte = featurize(s["test"][0], stoi, emb)
    with torch.no_grad():
        p = torch.softmax(model(torch.tensor((xte - mu) / sd, dtype=torch.float32)), -1)[:, 1].numpy()
    print(f"test macro-F1 (reference; final numbers come from src.benchmark): "
          f"{macro_f1(s['test'][1], (p >= t).astype(int)):.4f}")


if __name__ == "__main__":
    main()
