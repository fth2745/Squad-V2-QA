# Squad-V2-QA
# ❓ DeBERTa-Based Extractive Question Answering (SQuAD v2 Style)

> A transformer-based QA model with DeBERTa and SQuAD v2 logic — span prediction meets answerability.

This repository contains a fully developed extractive question answering system that leverages the power of transformer models to understand and accurately answer natural language queries based on context. The model is based on the SQuAD v2 dataset structure, which includes both answerable and unanswerable questions.

---

## 🔍 Project Overview

This QA system focuses on two primary objectives:

1. **Span Prediction:** Identifying the exact start and end token positions of the answer within the given context.
2. **Answerability Classification:** Determining whether the question can be answered based on the provided passage (i.e., handling `[NO_ANS]` cases).

The model is currently built using `microsoft/deberta-v3-large` and is designed to be lightweight, extensible, and adaptable for multilingual or domain-specific datasets.

---

## ⚙️ Technologies Used (So Far)

| Component           | Description                                                                 |
|---------------------|-----------------------------------------------------------------------------|
| `DeBERTa-v3-Large`   | Transformer encoder for deep contextual representations                    |
| `PyTorch`           | Core framework for model training, evaluation, and optimization             |
| `Huggingface Transformers` | Tokenization, pretrained model loading, and architecture utilities   |
| `AMP (Mixed Precision)` | Faster and more memory-efficient training with `torch.cuda.amp`         |
| `CrossEntropyLoss`  | Used for start/end span loss and classification output                      |
| `QADataset` Class   | Custom dataset class supporting overflowed tokens, offset mapping, etc.     |
| `Evaluation`        | Exact Match (EM), F1 Score, classification accuracy, and false match logging|
| `BERTScore`        | Semantic similarity metric for more nuanced evaluation of predicted answers |
                       

---

## 📂 Dataset

- **Input Format:** A custom SQuAD v2-style dataset with summarized contexts and manually annotated answers.
- **Columns:**  
  - `context_summarized` :  
  - `question_dl`  
  - `answer_dl`  
  - `answer_start_context`  
- **Unanswerable Questions:** Represented with `[NO_ANS]` tag and managed through binary classification logic.

## Preprocessing Steps

To optimize model performance and improve data quality, the following fundamental preprocessing steps are applied (excluding punctuation removal and lemmatization):
Lowercasing: All text is converted to lowercase to reduce the model’s sensitivity to case variations.
Whitespace Normalization: Excessive spaces are reduced to single spaces to standardize the text.
Special Character Cleaning: Unnecessary symbols and control characters within the text are removed.
Tokenization: Context and question texts are segmented using appropriate tokenization strategies.
Context and Question Length Optimization: Texts are split according to maximum sequence length and document stride parameters, with careful handling of overflow tokens.

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
- `BERTScore F1` to assess semantic equivalence between predicted and ground-truth answers

---

## 📊 Evaluation Metrics: Why EM, F1, and BERTScore F1 Are All Necessary
In extractive question answering tasks, evaluating model performance solely based on exact textual matches is often insufficient. Therefore, multiple metrics are used to capture various levels of correctness, from strict token-level overlap to deeper semantic alignment.

Exact Match (EM):
EM measures the percentage of predictions that match the ground truth answer exactly, after normalization (e.g., lowercasing, punctuation removal). This metric reflects the model's ability to identify the precise answer span without deviation.

F1 Score (Token-Level):
The F1 score calculates the harmonic mean of precision and recall between the predicted and ground-truth tokens. It rewards partial overlaps, making it more flexible than EM. This is especially important when the predicted answer is close to the ground truth but not identical (e.g., minor token differences, slight phrasing variations).

BERTScore F1 (Semantic-Level):
BERTScore evaluates the semantic similarity between the predicted and ground-truth answers using contextual embeddings from transformer-based models (e.g., BERT). Instead of relying on surface-form overlap, BERTScore compares token-level embeddings in context.
The F1 variant of BERTScore provides a robust measurement of how semantically equivalent the predicted answer is to the reference. It is especially useful in cases where the predicted answer uses synonyms, paraphrasing, or different word orders that preserve the same meaning.

🔍 Why is it important to combine these metrics?

EM captures the model’s ability to locate the exact answer string.

F1 accounts for partial correctness at the token level.

BERTScore F1 measures semantic fidelity, regardless of token or phrase surface form.

> In SQuAD v2-style settings, where the dataset includes both answerable and unanswerable questions, relying on only one metric (e.g., EM or F1) can lead to misleading conclusions. BERTScore complements traditional metrics by providing a semantic lens, which is crucial > for evaluating real-world generalization and robustness of the model.
---

### Why Focus on EM and F1 Alongside Loss?

While the **training loss** (e.g., CrossEntropyLoss) indicates how well the model is optimizing its parameters to minimize prediction errors, it does not directly measure the quality or usefulness of the predicted answers in a real-world sense.

- **Loss reflects model confidence and error at a token/prediction level**, but may not correlate perfectly with exact or partial correctness of answers.
- **EM and F1 measure the end-goal performance**: how accurately the model finds the correct answer span and whether it produces meaningful answers.
- Tracking EM and F1 during validation gives a clearer picture of real-world effectiveness and helps to detect overfitting or underfitting beyond loss trends.
- In QA tasks, a model can have low loss but still produce answers that do not match ground truth spans well; hence, EM and F1 are crucial metrics to evaluate final performance.

Therefore, alongside loss, focusing on EM and F1 ensures a more comprehensive and practical evaluation of the model’s ability to answer questions correctly and meaningfully.

---

## 📈 Example Metrics and Hyperparameters
Test Set
Validation Loss: 0.4694 | Exact Match: 74.57 | F1: 84.06 | BERTScore F1: 95.98 

Classification Report (Answerability - binary):
              precision    recall  f1-score   support

   No Answer       1.00      0.89      0.94      5953
  Has Answer       0.90      1.00      0.95      5888
    accuracy                           0.94     11841
   macro avg       0.95      0.94      0.94     11841
weighted avg       0.95      0.94      0.94     11841

Hyperparameters
Model: microsoft/deberta-v3-large
Maximum Sequence Length: 256
Document Stride: 128
Batch Size: 32
Learning Rate: 3e-5
Epochs: 15
Early Stopping Patience: 3
Optimizer: AdamW with Lookahead
Mixed Precision Training: Enabled (AMP)

---

## 💡 Why This Project?

This project is for anyone curious about:

- How modern QA systems handle **unanswerable questions**
- Building **lightweight yet powerful extractive QA** models
- Working with **transformers and token-level classification**
- Exploring **hybrid evaluation strategies** (exact match + answerability)



