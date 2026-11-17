"""Efficient model (technique area D): export the fine-tuned BERTurk to ONNX (FP32) and quantize it
to INT8 with ONNX Runtime dynamic quantization (weights INT8, activations quantized on the fly).

Run after train_bert:  python -m src.export_onnx
Produces models/berturk_onnx/{model_fp32.onnx, model_int8.onnx, tokenizer files}
"""
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import BERT_FP32_ONNX, BERT_HF_DIR, BERT_INT8_ONNX, BERT_MAX_LEN, BERT_ONNX_DIR
from .data import load_splits
from .normalize import for_bert
from .utils import load_threshold, macro_f1

INPUTS = ["input_ids", "attention_mask", "token_type_ids"]


class LogitsOnly(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.m(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids).logits


def export_fp32(model, tok, out_path):
    enc = tok(["örnek bir mesaj", "ikinci daha uzun bir örnek mesaj"], padding=True, return_tensors="pt")
    args = tuple(enc[k] for k in INPUTS)
    wrapped = LogitsOnly(model).eval()
    try:  # torch.export-based exporter (default since PyTorch 2.9)
        batch, seq = torch.export.Dim("batch", max=1024), torch.export.Dim("seq", max=512)
        torch.onnx.export(wrapped, args, str(out_path), input_names=INPUTS, output_names=["logits"],
                          dynamic_shapes={k: {0: batch, 1: seq} for k in INPUTS}, dynamo=True,
                          external_data=False)
        print("exported with the torch.export (dynamo) ONNX exporter")
    except Exception as e:  # older torch: classic TorchScript exporter
        print(f"dynamo exporter failed ({type(e).__name__}: {e}); using the TorchScript exporter")
        torch.onnx.export(wrapped, args, str(out_path), input_names=INPUTS, output_names=["logits"],
                          dynamic_axes={k: {0: "batch", 1: "seq"} for k in INPUTS} | {"logits": {0: "batch"}},
                          opset_version=17, do_constant_folding=True, dynamo=False)


def quantize_int8(fp32_path, int8_path, work_dir):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    src = fp32_path
    try:  # recommended pre-processing (shape inference + graph optimization)
        from onnxruntime.quantization.shape_inference import quant_pre_process
        pre = work_dir / "model_fp32_pre.onnx"
        quant_pre_process(str(fp32_path), str(pre), skip_symbolic_shape=True)
        src = pre
    except Exception as e:
        print(f"pre-processing skipped: {e}")
    quantize_dynamic(str(src), str(int8_path), weight_type=QuantType.QInt8)


def onnx_probs(path, tok, texts, bs=64):
    import onnxruntime as ort
    sess = ort.InferenceSession(Path(path).read_bytes(), providers=["CPUExecutionProvider"])
    names = {i.name for i in sess.get_inputs()}
    out = []
    for i in range(0, len(texts), bs):
        enc = tok([for_bert(t) for t in texts[i:i + bs]], truncation=True, max_length=BERT_MAX_LEN,
                  padding=True, return_tensors="np")
        logits = sess.run(None, {k: enc[k].astype(np.int64) for k in INPUTS if k in names})[0]
        e = np.exp(logits - logits.max(-1, keepdims=True))
        out.append(e[:, 1] / e.sum(-1))
    return np.concatenate(out)


def main():
    tok = AutoTokenizer.from_pretrained(BERT_HF_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(BERT_HF_DIR, attn_implementation="eager").eval()

    # ONNX's C++ file I/O cannot open non-ASCII Windows paths (e.g. "Masaüstü"), so all ONNX files
    # are produced in an ASCII temp folder and moved into place afterwards.
    BERT_ONNX_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="onnx_export_") as tmp:
        tmp = Path(tmp)
        export_fp32(model, tok, tmp / "model_fp32.onnx")
        quantize_int8(tmp / "model_fp32.onnx", tmp / "model_int8.onnx", tmp)
        for name, dst in (("model_fp32.onnx", BERT_FP32_ONNX), ("model_int8.onnx", BERT_INT8_ONNX)):
            shutil.move(str(tmp / name), str(dst))
        leftovers = [p.name for p in tmp.iterdir()]
        assert not any(n.endswith(".data") for n in leftovers), f"unexpected external data files: {leftovers}"
    tok.save_pretrained(BERT_ONNX_DIR)
    thr = BERT_HF_DIR.parent / "thresholds.json"
    if thr.exists():
        shutil.copy(thr, BERT_ONNX_DIR / "thresholds.json")

    # sanity check 1: ONNX FP32 must reproduce PyTorch logits
    sample = ["@user siktir git ezik", "Helal olsun başkanım süper olmuş", "bu maç çok kötüydü ya"]
    enc = tok([for_bert(t) for t in sample], padding=True, return_tensors="pt")
    with torch.no_grad():
        pt = torch.softmax(model(**enc).logits, -1)[:, 1].numpy()
    ox = onnx_probs(BERT_FP32_ONNX, tok, sample)
    print(f"PyTorch probs {np.round(pt, 4)}  |  ONNX FP32 probs {np.round(ox, 4)}")
    assert np.allclose(pt, ox, atol=1e-3), "ONNX export does not match PyTorch!"

    # sanity check 2: the INT8 file must be much smaller
    mb = lambda p: p.stat().st_size / 2 ** 20
    print(f"size FP32={mb(BERT_FP32_ONNX):.1f} MB   INT8={mb(BERT_INT8_ONNX):.1f} MB")
    assert mb(BERT_INT8_ONNX) < 0.5 * mb(BERT_FP32_ONNX), "quantization did not shrink the model"

    # reference test scores (final numbers + latency: python -m src.benchmark on the target CPU)
    s = load_splits()
    t = load_threshold("bert")
    for name, path in (("FP32", BERT_FP32_ONNX), ("INT8", BERT_INT8_ONNX)):
        p = onnx_probs(path, tok, s["test"][0])
        print(f"ONNX {name}: test macro-F1 = {macro_f1(s['test'][1], (p >= t).astype(int)):.4f}")


if __name__ == "__main__":
    main()
