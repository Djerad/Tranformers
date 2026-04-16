🧠 Named Entity Recognition (NER) with BERT Fine-Tuning

This project implements a Named Entity Recognition (NER) system using a fine-tuned BERT model on the WikiANN dataset.
It identifies entities in text such as:

👤 Persons (PER)
🏢 Organizations (ORG)
📍 Locations (LOC)
🚀 Project Overview

We fine-tune a pre-trained bert-base-cased model to perform token-level classification for NER tasks using HuggingFace Transformers.

The pipeline includes:

Dataset loading (WikiANN)
Tokenization with label alignment
BERT fine-tuning
Evaluation using seqeval
Inference pipeline
Model saving for deployment
📊 Dataset

We use the WikiANN dataset (English version):

Format: IOB2 tagging scheme
Labels:
O → Outside entity
B-PER, I-PER → Person
B-ORG, I-ORG → Organization
B-LOC, I-LOC → Location
⚙️ Installation
pip install transformers datasets evaluate seqeval accelerate
📁 Project Structure
.
├── train.py (or notebook)
├── model/
├── tokenizer/
└── README.md
🧠 Model Architecture
Base Model: bert-base-cased
Task: Token Classification (NER)
Number of Labels: 7
Training Strategy:
WordPiece tokenization
Label alignment (-100 for sub-tokens)
AdamW optimizer
FP16 training (GPU optimized)
✂️ Key Step: Tokenization & Label Alignment

Since BERT splits words into sub-tokens:

Example:

"Hawthorne" → ["Haw", "##thorne"]

We:

Assign the label to the first token
Assign -100 to remaining sub-tokens (ignored in loss)
🏋️ Training Details
Epochs: 3
Learning Rate: 2e-5
Batch Size: dynamic (data collator)
Warmup Ratio: 0.1
GPU: T4 (Google Colab)
📈 Results
Validation Performance
Accuracy: ~93%
F1 Score: ~83%
Test Performance
Precision: 0.8190
Recall:    0.8465
F1 Score:   0.8325
🔍 Inference Example
text = "Elon Musk founded SpaceX in Hawthorne, California."

Output:

[PER] Elon Musk
[ORG] SpaceX
[LOC] Hawthorne, California
💾 Save & Load Model
Save model:
model.save_pretrained("./ner-model")
tokenizer.save_pretrained("./ner-model")
Load model:
from transformers import pipeline

ner = pipeline("ner", model="./ner-model", tokenizer="./ner-model", aggregation_strategy="simple")
📦 Technologies Used
HuggingFace Transformers 🤗
PyTorch 🔥
Datasets Library 📊
SeqEval 📏
Google Colab GPU 🚀