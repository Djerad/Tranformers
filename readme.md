# 📰 Text Summarizer — BART + Gradio

Summarize any long article or news piece in seconds using
**facebook/bart-large-cnn** (or swap to T5) with a clean local web UI.

---

## 🗂 Project structure

```
text-summarizer/
├── app.py            ← main application (model + Gradio UI)
├── requirements.txt  ← Python dependencies
└── README.md
```

---

## ⚙️ Installation guide

### Step 1 — Prerequisites

Make sure you have **Python 3.9 or higher** installed.

```bash
python --version        # should print 3.9+
```

If not, download it from https://python.org

---

### Step 2 — Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

> ⏱ This installs PyTorch, Hugging Face Transformers, and Gradio.
> May take 2–5 minutes depending on your connection.

---

### Step 4 — Run the app

```bash
python app.py
```

On **first run**, the BART model (~1.6 GB) is downloaded automatically
from the Hugging Face Hub and cached in `~/.cache/huggingface/`.
Subsequent runs start in seconds.

The terminal will print:
```
⏳  Loading model 'facebook/bart-large-cnn' …
✅  Model ready!
Running on local URL:  http://0.0.0.0:7860
```

Your browser opens automatically at **http://localhost:7860**

---

## 🎛 How to use

1. Paste any article (30+ words) into the left panel.
2. Pick a **summary length**: Short / Medium / Long.
3. Adjust **beam search width** (4 is the sweet spot).
4. Click **▸ Summarize** (or press Enter in the text box).
5. The summary appears on the right with a word-count and compression ratio.

Two ready-made example articles are included — click them to try instantly.

---

## 🔀 Switch to T5

Open `app.py` and change line 11:

```python
# BART (default — best for news)
MODEL_ID = "facebook/bart-large-cnn"

# T5 base (~240 MB — much lighter)
MODEL_ID = "t5-base"

# Flan-T5 Large — better quality T5 variant
MODEL_ID = "google/flan-t5-large"
```

No other changes needed.

---

## 💡 Tips

| Want faster runs?      | Lower the beam width slider to 1–2.            |
|------------------------|------------------------------------------------|
| Have an NVIDIA GPU?    | It's used automatically — ~5× faster.         |
| Want a public URL?     | Set `share=True` in `demo.launch()` in app.py.|
| Offline after 1st run? | Model is cached; no internet needed.           |

---

## 📦 Dependencies explained

| Package        | Why                                          |
|----------------|----------------------------------------------|
| torch          | Neural network backend (CPU or CUDA)         |
| transformers   | BART / T5 model + `pipeline()` abstraction   |
| sentencepiece  | Tokenizer for T5 models                      |
| accelerate     | Faster model loading                         |
| gradio         | Web UI — zero HTML/JS needed                 |

---

## 🧠 What you'll learn from this project

- `pipeline("summarization", model=...)` — the simplest inference API in Transformers
- BART architecture — encoder-decoder, pre-trained on CNN/DailyMail news
- T5 architecture — text-to-text framing for any NLP task
- Beam search — how models generate high-quality sequences
- `min_length` / `max_length` / `num_beams` — key generation parameters
- Gradio Blocks API — building interactive ML UIs in pure Python