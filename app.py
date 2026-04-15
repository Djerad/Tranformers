import gradio as gr
from transformers import pipeline
import torch

# ── Model loading ────────────────────────────────────────────────────────────
# Uses facebook/bart-large-cnn by default (best for news/articles).
# Swap MODEL_ID to "t5-base" or "google/flan-t5-large" if you prefer T5.

MODEL_ID = "facebook/bart-large-cnn"

device = 0 if torch.cuda.is_available() else -1   # GPU if available, else CPU

print(f"⏳  Loading model '{MODEL_ID}' … (first run downloads ~1.6 GB)")
summarizer = pipeline(
    "summarization",
    model=MODEL_ID,
    device=device,
)
print("✅  Model ready!")

# ── Summarise function ────────────────────────────────────────────────────────
def summarize(text: str, length_preset: str, num_beams: int) -> tuple[str, str]:
    """
    Returns (summary_text, info_string).
    length_preset controls min/max output tokens.
    num_beams controls beam search width (higher = better quality, slower).
    """
    text = text.strip()
    if not text:
        return "", "⚠️  Please paste some text first."

    word_count = len(text.split())

    if word_count < 30:
        return "", "⚠️  Text is too short to summarize (need at least 30 words)."

    # Map preset → (min_length, max_length)
    presets = {
        "Short  (~1–2 sentences)":  (30,  80),
        "Medium (~3–5 sentences)":  (80,  180),
        "Long   (~6–10 sentences)": (180, 350),
    }
    min_len, max_len = presets[length_preset]

    # Guard: max_length must be < input token count
    max_len = min(max_len, word_count - 1)
    min_len = min(min_len, max_len - 5)

    result = summarizer(
        text,
        min_length=min_len,
        max_length=max_len,
        num_beams=num_beams,
        early_stopping=True,
        no_repeat_ngram_size=3,   # avoids repetitive phrases
    )

    summary = result[0]["summary_text"]
    summary_words = len(summary.split())
    compression = round((1 - summary_words / word_count) * 100, 1)

    info = (
        f"📄  Input: {word_count} words  →  "
        f"📝  Summary: {summary_words} words  "
        f"(compressed {compression}%)"
    )
    return summary, info


# ── Example articles (for the Examples row) ──────────────────────────────────
EXAMPLES = [
    [
        """Artificial intelligence (AI) is transforming industries across the globe, from healthcare
        and finance to transportation and entertainment. In healthcare, AI algorithms can now detect
        certain cancers with greater accuracy than experienced radiologists. Financial institutions
        use machine learning models to detect fraudulent transactions in real time, saving billions
        annually. Autonomous vehicles rely on deep neural networks to perceive their surroundings
        and make split-second decisions. Meanwhile, generative AI tools like large language models
        have begun reshaping creative work, software development, and customer service. Despite
        these advances, concerns around bias, privacy, job displacement, and the concentration of
        power in a handful of large technology companies continue to spark debate among
        policymakers, ethicists, and researchers worldwide.""",
        "Medium (~3–5 sentences)",
        4,
    ],
    [
        """The James Webb Space Telescope (JWST) has delivered its most detailed image of the Crab
        Nebula to date, offering astronomers an unprecedented view of the remnants of a massive
        stellar explosion observed nearly a thousand years ago. The image reveals intricate
        structures within the nebula — glowing gas filaments, a rapidly spinning neutron star at
        its centre, and wisps of charged particles accelerated to near the speed of light. Scientists
        believe that studying the Crab Nebula in such detail will help refine models of how
        supernovae seed the cosmos with heavy elements essential for planetary formation and life.
        The new data, captured using JWST's Near-Infrared Camera and Mid-Infrared Instrument,
        represents a milestone in observational astrophysics and will be freely available to
        researchers around the world.""",
        "Short  (~1–2 sentences)",
        4,
    ],
]

# ── Gradio UI ─────────────────────────────────────────────────────────────────
THEME = gr.themes.Base(
    primary_hue="slate",
    secondary_hue="blue",
    neutral_hue="zinc",
    font=gr.themes.GoogleFont("IBM Plex Mono"),
).set(
    body_background_fill="#0f1117",
    body_text_color="#e2e8f0",
    block_background_fill="#1a1d27",
    block_border_color="#2d3148",
    input_background_fill="#12141e",
    button_primary_background_fill="#3b5bdb",
    button_primary_background_fill_hover="#4c6ef5",
    button_primary_text_color="#ffffff",
)

CSS = """
/* ── global ── */
body { background: #0f1117; }

/* header */
#header { text-align: center; padding: 2rem 0 1rem; }
#header h1 {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #e2e8f0;
    letter-spacing: -1px;
    margin-bottom: 0.3rem;
}
#header p {
    color: #64748b;
    font-size: 0.95rem;
    font-family: 'IBM Plex Mono', monospace;
}
#model-badge {
    display: inline-block;
    background: #1e2235;
    border: 1px solid #3b5bdb44;
    color: #7c8cf8;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.78rem;
    padding: 3px 10px;
    border-radius: 4px;
    margin-top: 0.5rem;
}

/* panels */
.panel-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: #475569;
    margin-bottom: 4px;
}

/* info bar */
#info-bar {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.8rem;
    color: #7c8cf8;
    min-height: 1.4rem;
    padding: 4px 0;
}

/* summarise button */
#run-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1rem;
    letter-spacing: 0.08em;
    height: 52px;
    border-radius: 6px;
}

/* clear button */
#clear-btn {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.85rem;
    color: #64748b;
    height: 52px;
}

/* textareas */
textarea {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.88rem !important;
    line-height: 1.7 !important;
    color: #cbd5e1 !important;
}

/* footer */
#footer {
    text-align: center;
    padding: 1.5rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    color: #334155;
}
"""

with gr.Blocks(theme=THEME, css=CSS, title="Text Summarizer") as demo:

    # ── Header ────────────────────────────────────────────────────────────────
    gr.HTML("""
        <div id="header">
            <h1>// TEXT SUMMARIZER</h1>
            <p>Paste any article or news piece — get a clean summary instantly.</p>
            <span id="model-badge">▸ facebook/bart-large-cnn &nbsp;|&nbsp; beam search inference</span>
        </div>
    """)

    # ── Main layout ───────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        with gr.Column(scale=5):
            gr.HTML('<div class="panel-label">① Input text</div>')
            input_text = gr.Textbox(
                placeholder="Paste a long article, news story, or any body of text here …",
                lines=18,
                max_lines=40,
                show_label=False,
                elem_id="input-box",
            )

        with gr.Column(scale=5):
            gr.HTML('<div class="panel-label">② Summary</div>')
            output_text = gr.Textbox(
                lines=18,
                max_lines=40,
                show_label=False,
                interactive=False,
                elem_id="output-box",
            )

    # ── Info bar ──────────────────────────────────────────────────────────────
    info_text = gr.Markdown("", elem_id="info-bar")

    # ── Controls ──────────────────────────────────────────────────────────────
    with gr.Row():
        length_preset = gr.Radio(
            choices=[
                "Short  (~1–2 sentences)",
                "Medium (~3–5 sentences)",
                "Long   (~6–10 sentences)",
            ],
            value="Medium (~3–5 sentences)",
            label="Summary length",
        )
        num_beams = gr.Slider(
            minimum=1,
            maximum=8,
            value=4,
            step=1,
            label="Beam search width  (higher = better quality, slower)",
        )

    with gr.Row():
        clear_btn = gr.Button("✕  Clear", variant="secondary", elem_id="clear-btn")
        run_btn   = gr.Button("▸  Summarize", variant="primary",  elem_id="run-btn",  scale=3)

    # ── Examples ──────────────────────────────────────────────────────────────
    gr.Examples(
        examples=EXAMPLES,
        inputs=[input_text, length_preset, num_beams],
        label="📋  Try an example",
        cache_examples=False,
    )

    # ── Footer ────────────────────────────────────────────────────────────────
    gr.HTML("""
        <div id="footer">
            built with 🤗 Transformers + Gradio &nbsp;|&nbsp;
            model: facebook/bart-large-cnn &nbsp;|&nbsp;
            runs on CPU or CUDA
        </div>
    """)

    # ── Events ────────────────────────────────────────────────────────────────
    run_btn.click(
        fn=summarize,
        inputs=[input_text, length_preset, num_beams],
        outputs=[output_text, info_text],
    )
    # Also trigger on Ctrl+Enter inside the input box
    input_text.submit(
        fn=summarize,
        inputs=[input_text, length_preset, num_beams],
        outputs=[output_text, info_text],
    )
    clear_btn.click(
        fn=lambda: ("", "", ""),
        outputs=[input_text, output_text, info_text],
    )


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",   # accessible on your local network too
        server_port=7860,
        share=False,             # set True to get a public gradio.live link
        inbrowser=True,          # auto-opens browser tab
    )