# Squad-V2-QA
# ❓ DeBERTa-Based Extractive Question Answering (SQuAD v2 Style)

> A transformer-based QA model with DeBERTa and SQuAD v2 logic — span prediction meets answerability.

![qa-model-diagram](./assets/qa_model_architecture.png)

This repository contains a **work-in-progress extractive question answering system** that leverages the power of **transformer models** to understand and answer natural language queries based on context. The model is inspired by the **SQuAD v2** dataset structure, which includes both **answerable and unanswerable** questions.

---

## 🔍 Project Overview

This QA system focuses on two primary objectives:

1. **Span Prediction:** Identifying the exact start and end token positions of the answer within the given context.
2. **Answerability Classification:** Determining whether the question can be answered based on the provided passage (i.e., handling `[NO_ANS]` cases).

The model is currently built using `microsoft/deberta-v3-base` and is designed to be lightweight, extensible, and adaptable for multilingual or domain-specific datasets.

---

## ⚙️ Technologies Used (So Far)

| Component            | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `DeBERTa-v3-base`   | Transformer encoder for deep contextual representations                    |
| `PyTorch`           | Core framework for model training, evaluation, and optimization             |
| `Huggingface Transformers` | Tokenization, pretrained model loading, and architecture utilities        |
| `AMP (Mixed Precision)` | Faster and more memory-efficient training with `torch.cuda.amp`           |
| `CrossEntropyLoss`  | Used for start/end span loss and classification output                      |
| `QADataset` Class   | Custom dataset class supporting overflowed tokens, offset mapping, etc.    |
| `Evaluation`        | Exact Match (EM), F1 Score, classification accuracy, and false match logging|

---

## 📂 Dataset

- **Input Format:** A custom SQuAD v2-style dataset with summarized contexts and manually annotated answers.
- **Columns:**  
  - `context_summarized`  
  - `question_dl`  
  - `answer_dl`  
  - `answer_start_context`  
- **Unanswerable Questions:** Represented with `[NO_ANS]` tag and managed through binary classification logic.

---

## 🧠 Model Architecture

