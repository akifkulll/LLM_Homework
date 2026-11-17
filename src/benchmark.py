"""Evaluate every model of the ladder on the SAME held-out test set (official OffensEval-TR test).

For each model: macro-F1, precision/recall of the offensive class, size on disk (MB) and latency
(ms per message, batch size 1, CPU, preprocessing included). Latency = mean over all 3,528 test
messages (>= 100 runs), after a warm-up.

Run:  python -m src.benchmark            (writes results/results.json, results.md, errors_*.csv)
      python -m src.benchmark --threads 1   (pin CPU threads for a fairer, more stable latency)
"""
import argparse
import csv
import json
import os
import platform
import time

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from .config import LATENCY_BUDGET_MS, RESULTS_DIR
from .data import load_splits


def cpu_name() -> str:
    try:
        if platform.system() == "Windows":
            import winreg
            k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            import subprocess
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    except Exception:
        pass
    return platform.processor() or "unknown CPU"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=0, help="limit CPU threads (0 = library default)")
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N test messages (smoke test)")
    args = ap.parse_args()
    if args.threads:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        import torch
        torch.set_num_threads(args.threads)

    from .models import load_models, size_mb
    models = load_models()
    if args.threads:
        import onnxruntime as ort
        from .models import OnnxBertModel
        for m in models.values():
            if isinstance(m, OnnxBertModel):
                o = ort.SessionOptions()
                o.intra_op_num_threads = args.threads
                m.sess = ort.InferenceSession(m.path.read_bytes(), o, providers=["CPUExecutionProvider"])

    x, y = load_splits()["test"]
    if args.limit:
        x, y = x[:args.limit], y[:args.limit]
    hw = f"{cpu_name()} | {os.cpu_count()} logical cores | {platform.system()} {platform.release()}"
    print(f"Test set: {len(x)} messages | hardware: {hw}\n")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for key, m in models.items():
        for t in x[:20]:                        # warm-up
            m.predict_proba(t)
        probs, times = [], []
        for t in x:
            t0 = time.perf_counter()
            probs.append(m.predict_proba(t))
            times.append((time.perf_counter() - t0) * 1000)
        probs, times = np.array(probs), np.array(times)
        pred = (probs >= m.threshold).astype(int)
        p, r, f, _ = precision_recall_fscore_support(y, pred, average=None, labels=[0, 1], zero_division=0)
        row = {
            "model": m.name, "key": key,
            "macro_f1": float(f.mean()), "accuracy": float(accuracy_score(y, pred)),
            "off_precision": float(p[1]), "off_recall": float(r[1]), "off_f1": float(f[1]),
            "size_mb": round(size_mb(m), 4),
            "latency_mean_ms": float(times.mean()), "latency_p50_ms": float(np.percentile(times, 50)),
            "latency_p95_ms": float(np.percentile(times, 95)), "runs": len(times),
            "threshold": m.threshold, "within_budget": bool(np.percentile(times, 95) <= LATENCY_BUDGET_MS),
        }
        rows.append(row)
        print(f"{m.name:34s} macro-F1={row['macro_f1']:.4f}  size={row['size_mb']:8.2f} MB  "
              f"latency mean={row['latency_mean_ms']:7.2f} ms  p95={row['latency_p95_ms']:7.2f} ms")
        # misclassified examples -> failure analysis slide
        with open(RESULTS_DIR / f"errors_{key}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["gold", "pred", "p_offensive", "text"])
            for t, g, pr, pp in zip(x, y, pred, probs):
                if g != pr:
                    w.writerow(["OFF" if g else "NOT", "OFF" if pr else "NOT", f"{pp:.3f}", t])

    out = {"hardware": hw, "test_size": len(x), "latency_budget_ms": LATENCY_BUDGET_MS,
           "threads": args.threads or "default", "results": rows}
    (RESULTS_DIR / "results.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    md = [f"Hardware: {hw} (threads: {args.threads or 'default'}) | test set: {len(x)} messages | "
          f"latency budget: {LATENCY_BUDGET_MS:.0f} ms (p95)\n",
          "| Model | Macro-F1 | OFF recall | Size (MB) | Latency mean (ms) | Latency p95 (ms) | Budget |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        size = f"{r['size_mb']:.1f}" if r["size_mb"] >= 1 else f"{r['size_mb']:.3f}"
        md.append(f"| {r['model']} | {r['macro_f1']:.4f} | {r['off_recall']:.4f} | {size} | "
                  f"{r['latency_mean_ms']:.2f} | {r['latency_p95_ms']:.2f} | {'✅' if r['within_budget'] else '❌'} |")
    (RESULTS_DIR / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\nSaved {RESULTS_DIR / 'results.md'} and results.json")


if __name__ == "__main__":
    main()
