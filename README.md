# Squad-V2-QA
# ❓ DeBERTa-Based Extractive Question Answering (SQuAD v2 Style)

> A transformer-based QA model with DeBERTa and SQuAD v2 logic — span prediction meets answerability.

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

        Input Pair (Question + Context)
                      ↓
            Tokenization (DeBERTa)
                      ↓
       Transformer Embeddings (hidden states)
                      ↓
        ┌─────────────────────────────┐
        │   QA Head (Start / End)     │
        │   Answerability Classifier  │
        └─────────────────────────────┘
                      ↓
      Prediction: Start, End, Answerable?


The model performs multi-head classification, simultaneously predicting:
- **Answer span (start, end tokens)**
- **Binary label for answerability**

---

## 🧪 Training & Evaluation Features

- `Exact Match (EM)` and `F1 Score` for span prediction
- `Classification Report` for answerability (Has Answer / No Answer)
- `Semantic Penalty` to penalize semantically inconsistent predictions
- `False Match Logging` to analyze partially correct answers (`false_matches_log.txt`)
- `Early Stopping` based on validation F1 performance

---

## 🚧 Planned Extensions & Experimental Ideas

This repository is an open research environment — multiple improvements and experiments are under consideration:

- [ ] **FGM (Fast Gradient Method):** Incorporate adversarial examples during training to improve model robustness
- [ ] **Teacher-Student Distillation:** Leverage stronger models (e.g., `deepset/deberta-v3-large`) as teachers for lighter student models
- [ ] **Curriculum Learning:** Train the model from easier to harder questions progressively
- [ ] **Confidence Thresholding:** Dynamically reject low-confidence answers in no-answer scenarios
- [ ] **Cross-Attention Fusion:** Integrate external semantic features (e.g., SBERT) alongside DeBERTa token embeddings
- [ ] **Attention Visualization:** Use gradient-based attention heatmaps to visualize model focus during inference

If you are interested in these topics, feel free to follow the project or contribute!

---

## 📊 Evaluation Metrics: Understanding EM and F1

In extractive question answering tasks, **Exact Match (EM)** and **F1 Score** are two fundamental evaluation metrics used to measure how well the model's predicted answers align with the ground truth answers.

- **Exact Match (EM):**  
  EM measures the percentage of predictions that match exactly with the reference answer span. It is a strict metric that only counts a prediction as correct if it perfectly matches the ground truth text (after normalization such as lowercasing and removing punctuation). This metric reflects the model’s ability to pinpoint the exact answer span.

- **F1 Score:**  
  F1 is the harmonic mean of precision and recall calculated at the token level between the predicted and ground truth answer spans. It provides a more forgiving and nuanced evaluation by rewarding partial overlaps. F1 is especially important when the model's predicted answer is close but not an exact match, which is common in natural language.

Together, these metrics provide complementary perspectives on model performance:

- **EM highlights exact correctness** — crucial in applications requiring precise answers.  
- **F1 captures partial correctness and relevance**, allowing some flexibility in phrasing.

In SQuAD v2-style datasets, which include both answerable and unanswerable questions, monitoring these metrics ensures the model not only finds accurate spans but also correctly identifies when no answer exists.

---

### Why Focus on EM and F1 Alongside Loss?

While the **training loss** (e.g., CrossEntropyLoss) indicates how well the model is optimizing its parameters to minimize prediction errors, it does not directly measure the quality or usefulness of the predicted answers in a real-world sense.

- **Loss reflects model confidence and error at a token/prediction level**, but may not correlate perfectly with exact or partial correctness of answers.
- **EM and F1 measure the end-goal performance**: how accurately the model finds the correct answer span and whether it produces meaningful answers.
- Tracking EM and F1 during validation gives a clearer picture of real-world effectiveness and helps to detect overfitting or underfitting beyond loss trends.
- In QA tasks, a model can have low loss but still produce answers that do not match ground truth spans well; hence, EM and F1 are crucial metrics to evaluate final performance.

Therefore, alongside loss, focusing on EM and F1 ensures a more comprehensive and practical evaluation of the model’s ability to answer questions correctly and meaningfully.

---

## 📈 Example Metrics (Dummy Snapshot)
Epoch 5/15
Validation Loss: 0.6309 | EM: 72.25 | F1: 83.28
Answerability Accuracy: 94.1%


---

## 💡 Why This Project?

This project is for anyone curious about:

- How modern QA systems handle **unanswerable questions**
- Building **lightweight yet powerful extractive QA** models
- Working with **transformers and token-level classification**
- Exploring **hybrid evaluation strategies** (exact match + answerability)



