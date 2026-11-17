"""Real-time chat moderation demo (Gradio).

Every message is labeled live (clean / offensive) with a confidence score; offensive words are
masked; the side panel logs the response time of every message and lets you switch models.
Default model: BERTurk INT8 (ONNX Runtime, CPU).

Run:  python app.py      (or: python run.py, which also prepares missing models first)
"""
import html
import time

import gradio as gr
import pandas as pd

from src.config import LATENCY_BUDGET_MS
from src.lexicon import Lexicon
from src.models import load_models
from src.normalize import for_classic

MODELS = load_models()
# Warm-up: ONNX Runtime / PyTorch are slow on the first calls (lazy init, memory allocation).
# The benchmark warms up too, so live latencies are comparable from the very first message.
for _m in MODELS.values():
    for _t in ("ısınma mesajı", "bu bir warm-up cümlesidir, biraz daha uzun bir örnek", "gg"):
        _m.predict_proba(_t)
NAME2KEY = {m.name: k for k, m in MODELS.items()}
DEFAULT = MODELS["bert_int8"].name if "bert_int8" in MODELS else list(NAME2KEY)[-1]
LEX = Lexicon()
LOG_COLS = ["#", "model", "label", "P(offensive)", "latency (ms)", f"≤{LATENCY_BUDGET_MS:.0f} ms"]
CMP_COLS = ["model", "label", "P(offensive)", "threshold", "latency (ms)"]

CSS = """
#chat {height: 460px; overflow-y: auto; padding: 12px; border-radius: 12px;
       background: var(--background-fill-secondary);}
.msg {max-width: 80%; margin: 8px 0; padding: 8px 12px; border-radius: 12px; line-height: 1.4;}
.clean {background: #e7f6ec; color: #11361f; border-left: 4px solid #2e9e5b;}
.off {background: #fdecec; color: #4a1010; border-left: 4px solid #d64545;}
.meta {font-size: 12px; color: #555; margin-top: 4px;}
"""


def mask(text: str) -> str:
    """Masks words that hit the lexicon; if the model flags a message without any lexicon word,
    the whole message is hidden."""
    out, hit = [], False
    for w in text.split():
        if any(LEX.token_hit(t) for t in for_classic(w).split()):
            core = w.strip(".,!?;:()\"'")
            out.append(w.replace(core, core[:1] + "*" * (len(core) - 1)) if core else w)
            hit = True
        else:
            out.append(w)
    return " ".join(out) if hit else "🚫 [message hidden: offensive content]"


def render(history):
    if not history:
        return "<div id='chat'><i>Type a message below…</i></div>"
    parts = []
    for h in history:
        cls = "off" if h["off"] else "clean"
        tag = "OFFENSIVE" if h["off"] else "clean"
        parts.append(
            f"<div class='msg {cls}'>{html.escape(h['shown'])}"
            f"<div class='meta'>{tag} · P(off)={h['p']:.2f} · {h['ms']:.1f} ms · {html.escape(h['model'])}</div></div>")
    return "<div id='chat'>" + "".join(reversed(parts)) + "</div>"


def send(msg, model_name, compare, history, log):
    msg = (msg or "").strip()
    if not msg:
        return render(history), pd.DataFrame(log, columns=LOG_COLS), gr.update(), "", history, log
    m = MODELS[NAME2KEY[model_name]]
    t0 = time.perf_counter()
    p = m.predict_proba(msg)
    ms = (time.perf_counter() - t0) * 1000
    off = p >= m.threshold
    history = history + [{"shown": mask(msg) if off else msg, "off": off, "p": p, "ms": ms, "model": m.name}]
    log = [[len(history), m.short, "OFFENSIVE" if off else "clean", round(p, 3), round(ms, 2),
            "✅" if ms <= LATENCY_BUDGET_MS else "❌"]] + log

    cmp = gr.update()
    if compare:
        rows = []
        for k, mm in MODELS.items():
            t0 = time.perf_counter()
            pp = mm.predict_proba(msg)
            rows.append([mm.name, "OFFENSIVE" if pp >= mm.threshold else "clean", round(pp, 3),
                         mm.threshold, round((time.perf_counter() - t0) * 1000, 2)])
        cmp = pd.DataFrame(rows, columns=CMP_COLS)
    return render(history), pd.DataFrame(log, columns=LOG_COLS), cmp, "", history, log


def clear():
    return render([]), pd.DataFrame(columns=LOG_COLS), pd.DataFrame(columns=CMP_COLS), [], []


GRADIO6 = int(gr.__version__.split(".")[0]) >= 6   # Gradio 6 moved `css` from Blocks() to launch()

with gr.Blocks(title="Turkish Chat Moderation", **({} if GRADIO6 else {"css": CSS})) as demo:
    gr.Markdown("## 🛡️ Real-Time Turkish Chat Moderation\n"
                f"Each message is classified live as **clean / offensive**. Latency budget: "
                f"**{LATENCY_BUDGET_MS:.0f} ms per message on CPU**.")
    history_s, log_s = gr.State([]), gr.State([])
    with gr.Row():
        with gr.Column(scale=3):
            chat = gr.HTML(render([]))
            with gr.Row():
                box = gr.Textbox(placeholder="Mesajını yaz ve Enter'a bas…", show_label=False, scale=5)
                btn = gr.Button("Send", variant="primary", scale=1)
            gr.Examples(["gg iyi oyundu herkese", "s1kt1r git ezikkkk", "a.m.k bu ne biçim hakem",
                         "seni bu maçta öldürürüm 😂", "sıkıntı yok kanka yarın oynarız",
                         "takımın en işe yaramaz oyuncusu sensin"], inputs=box)
        with gr.Column(scale=2):
            model_dd = gr.Dropdown(list(NAME2KEY), value=DEFAULT, label="Model")
            compare_cb = gr.Checkbox(value=True, label="Also compare all models on each message")
            log_df = gr.Dataframe(pd.DataFrame(columns=LOG_COLS), label="Response-time log", interactive=False)
            cmp_df = gr.Dataframe(pd.DataFrame(columns=CMP_COLS), label="All models on the last message",
                                  interactive=False)
            clear_btn = gr.Button("Clear chat")

    outs = [chat, log_df, cmp_df, box, history_s, log_s]
    ins = [box, model_dd, compare_cb, history_s, log_s]
    box.submit(send, ins, outs)
    btn.click(send, ins, outs)
    clear_btn.click(clear, None, [chat, log_df, cmp_df, history_s, log_s])

def launch():
    demo.launch(**({"css": CSS} if GRADIO6 else {}))


if __name__ == "__main__":
    launch()
