Hardware: 12th Gen Intel(R) Core(TM) i5-12500H | 16 logical cores | Windows 11 (threads: default) | test set: 3528 messages | latency budget: 50 ms (p95)

| Model | Macro-F1 | OFF recall | Size (MB) | Latency mean (ms) | Latency p95 (ms) | Budget |
|---|---|---|---|---|---|---|
| Lexicon + edit distance (rules) | 0.6669 | 0.3324 | 0.001 | 3.67 | 10.00 | ✅ |
| TF-IDF + LogReg | 0.7715 | 0.5698 | 2.0 | 1.44 | 1.92 | ✅ |
| word2vec + FFN (own) | 0.6843 | 0.4721 | 11.5 | 0.17 | 0.30 | ✅ |
| BERTurk FP32 (ONNX) | 0.8116 | 0.6997 | 423.0 | 19.13 | 34.75 | ✅ |
| BERTurk INT8 (ONNX, quantized) | 0.8102 | 0.6872 | 106.8 | 10.09 | 18.08 | ✅ |
