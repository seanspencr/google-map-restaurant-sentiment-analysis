"""
fine_tune_transformer.py
========================
Fine‑tunes a DistilBERT model for 3‑class sentiment classification
on restaurant reviews.

Data Transformation (WordPiece tokenisation via HuggingFace):
    • input_ids        – WordPiece token indices
    • attention_mask   – 1 for real tokens, 0 for padding
    • token_type_ids   – segment IDs (all zeros for single‑sentence;
                         generated for completeness – DistilBERT ignores them)

Model  : distilbert-base-uncased  →  DistilBertForSequenceClassification (3 classes)
Eval   : Accuracy · Precision · Recall · F1
Power  : CodeCarbon – single classification task
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from codecarbon import EmissionsTracker

# Re‑use the shared preprocessing pipeline
from preprocess import load_and_preprocess

warnings.filterwarnings("ignore")

# ── Constants ───────────────────────────────────────────────────────────────
BASE_DIR       = os.path.dirname(__file__)
PLOT_DIR       = os.path.join(BASE_DIR, "plots")
OUTPUT_DIR     = os.path.join(BASE_DIR, "distilbert_output")
MODEL_NAME     = "distilbert-base-uncased"
MAX_LEN        = 256
BATCH_SIZE     = 16
EPOCHS         = 5
LEARNING_RATE  = 2e-5
SEED           = 42

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Using device: {DEVICE}")


# ────────────────────────────────────────────────────────────────────────────
# Custom PyTorch Dataset
# ────────────────────────────────────────────────────────────────────────────
class SentimentDataset(Dataset):
    """Wraps HuggingFace tokeniser output + integer labels."""

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone() for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


# ────────────────────────────────────────────────────────────────────────────
# Metric computation callback for Trainer
# ────────────────────────────────────────────────────────────────────────────
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="weighted", zero_division=0
    )
    acc = accuracy_score(labels, preds)
    return {
        "accuracy":  round(acc, 4),
        "precision": round(precision, 4),
        "recall":    round(recall, 4),
        "f1":        round(f1, 4),
    }


# ════════════════════════════════════════════════════════════════════════════
#  MAIN PIPELINE
# ════════════════════════════════════════════════════════════════════════════

# ────────────────────────────────────────────────────────────────────────────
# STEP 1 – Load & Preprocess Data
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Loading & Preprocessing Data")
print("=" * 60)

X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()

# ────────────────────────────────────────────────────────────────────────────
# STEP 2 – Data Transformation
#   WordPiece tokenisation · attention masks · segment IDs
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 – Data Transformation (WordPiece Tokenisation)")
print("=" * 60)

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

def tokenize_texts(texts):
    """
    Tokenise a list of strings and return a dict of tensors:
        input_ids, attention_mask, token_type_ids
    """
    return tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=True,      # segment IDs
    )


train_encodings = tokenize_texts(X_train["reviews"].tolist())
val_encodings   = tokenize_texts(X_val["reviews"].tolist())
test_encodings  = tokenize_texts(X_test["reviews"].tolist())

# ── Quick peek at tokenised output ──
print(f"[INFO] Tokeniser          : {MODEL_NAME} (WordPiece)")
print(f"[INFO] Max sequence length: {MAX_LEN}")
print(f"[INFO] Keys returned      : {list(train_encodings.keys())}")
print(f"[INFO] input_ids shape    : {train_encodings['input_ids'].shape}")
print(f"[INFO] attention_mask     : {train_encodings['attention_mask'].shape}")
print(f"[INFO] token_type_ids     : {train_encodings['token_type_ids'].shape}")
print()

# Show an example
print("[INFO] Example tokenisation (first review):")
print(f"  Original text  : {X_train['reviews'].iloc[0][:120]}…")
print(f"  input_ids[:20] : {train_encodings['input_ids'][0][:20].tolist()}")
print(f"  attn_mask[:20] : {train_encodings['attention_mask'][0][:20].tolist()}")
print(f"  seg_ids[:20]   : {train_encodings['token_type_ids'][0][:20].tolist()}")
print()
print("[NOTE] DistilBERT was designed without segment‑ID input; "
      "token_type_ids are generated here for completeness but the model "
      "ignores them during forward pass.")
print()

# ── Label encoding (string → integer) ──
label_encoder = LabelEncoder()
label_encoder.fit(["negative", "neutral", "positive"])   # fixed order

y_train_enc = label_encoder.transform(y_train)
y_val_enc   = label_encoder.transform(y_val)
y_test_enc  = label_encoder.transform(y_test)

print(f"[INFO] Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
print()

# ── Build Dataset objects ──
# Remove token_type_ids before creating datasets because DistilBERT's
# forward() does not accept them.  We keep only input_ids + attention_mask.
def _strip_segment_ids(enc):
    return {k: v for k, v in enc.items() if k != "token_type_ids"}

train_dataset = SentimentDataset(_strip_segment_ids(train_encodings), y_train_enc)
val_dataset   = SentimentDataset(_strip_segment_ids(val_encodings),   y_val_enc)
test_dataset  = SentimentDataset(_strip_segment_ids(test_encodings),  y_test_enc)

print(f"[INFO] Train dataset : {len(train_dataset)} samples")
print(f"[INFO] Val   dataset : {len(val_dataset)} samples")
print(f"[INFO] Test  dataset : {len(test_dataset)} samples")
print()

# ────────────────────────────────────────────────────────────────────────────
# STEP 3 – Model Setup & Fine‑Tuning
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 – Fine‑Tuning DistilBERT")
print("=" * 60)

model = DistilBertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
)

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
    logging_steps=10,
    seed=SEED,
    report_to="none",           # disable wandb / mlflow logging
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
)

print("[TRAIN] Starting fine‑tuning …\n")
train_result = trainer.train()
print()
print(f"[TRAIN] Training complete.")
print(f"[TRAIN] Total steps : {train_result.global_step}")
print(f"[TRAIN] Train loss  : {train_result.training_loss:.4f}")
print()

# ── Plot training log ──
log_history = trainer.state.log_history

# Extract epoch‑level metrics from logs
epoch_logs = [l for l in log_history if "eval_loss" in l]
if epoch_logs:
    epochs_list  = [l["epoch"] for l in epoch_logs]
    eval_loss    = [l["eval_loss"] for l in epoch_logs]
    eval_acc     = [l.get("eval_accuracy", 0) for l in epoch_logs]
    eval_f1      = [l.get("eval_f1", 0) for l in epoch_logs]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs_list, eval_loss, marker="o", label="Val Loss")
    axes[0].set_title("Validation Loss", fontsize=14, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()

    axes[1].plot(epochs_list, eval_acc, marker="o", label="Val Accuracy")
    axes[1].plot(epochs_list, eval_f1,  marker="s", label="Val F1")
    axes[1].set_title("Validation Metrics", fontsize=14, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].legend()

    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, "distilbert_training_curves.png"), dpi=150)
    plt.show()
    print(f"[INFO] Training curves saved to {PLOT_DIR}/distilbert_training_curves.png\n")

# ────────────────────────────────────────────────────────────────────────────
# STEP 4 – Evaluation on Test Set
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4 – Evaluation on Test Set")
print("=" * 60)

predictions = trainer.predict(test_dataset)
y_pred_idx  = np.argmax(predictions.predictions, axis=-1)

y_pred_labels = label_encoder.inverse_transform(y_pred_idx)
y_true_labels = label_encoder.inverse_transform(y_test_enc)

acc  = accuracy_score(y_true_labels, y_pred_labels)
prec = precision_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
rec  = recall_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
f1   = f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)

print(f"\n{'─' * 40}")
print(f"  DistilBERT (fine‑tuned)")
print(f"{'─' * 40}")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print()
print(classification_report(y_true_labels, y_pred_labels, zero_division=0))

# ── Confusion Matrix ──
cm = confusion_matrix(y_true_labels, y_pred_labels,
                      labels=["positive", "neutral", "negative"])
disp = ConfusionMatrixDisplay(cm, display_labels=["positive", "neutral", "negative"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues")
ax.set_title("Confusion Matrix – DistilBERT", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "cm_distilbert.png"), dpi=150)
plt.show()
print(f"[INFO] Confusion matrix saved to {PLOT_DIR}/cm_distilbert.png\n")

# ── Summary table ──
results_df = pd.DataFrame([{
    "Model":     "DistilBERT (fine‑tuned)",
    "Accuracy":  round(acc, 4),
    "Precision": round(prec, 4),
    "Recall":    round(rec, 4),
    "F1-Score":  round(f1, 4),
}])

print("EVALUATION SUMMARY")
print("=" * 60)
print(results_df.to_string(index=False))
print()

# ────────────────────────────────────────────────────────────────────────────
# STEP 5 – Power Consumption (CodeCarbon) – single classification task
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5 – Power Consumption (CodeCarbon)")
print("=" * 60)

# Prepare a single sample (keep batch dimension)
single_input = {
    "input_ids":      test_encodings["input_ids"][0:1].to(DEVICE),
    "attention_mask":  test_encodings["attention_mask"][0:1].to(DEVICE),
}

model.eval()
model.to(DEVICE)

tracker = EmissionsTracker(
    project_name="single_predict_DistilBERT",
    measure_power_secs=0.1,
    log_level="error",
    save_to_file=False,
)

tracker.start()
with torch.no_grad():
    _ = model(**single_input)
emissions = tracker.stop()  # kg CO₂

energy_kwh = tracker.final_emissions_data.energy_consumed  # kWh

print(f"\n[DistilBERT]  Energy: {energy_kwh:.10f} kWh  |  CO₂: {emissions:.10f} kg")

energy_df = pd.DataFrame([{
    "Model":              "DistilBERT (fine‑tuned)",
    "Energy (kWh)":       f"{energy_kwh:.10f}",
    "CO₂ Emissions (kg)": f"{emissions:.10f}",
}])
print()
print(energy_df.to_string(index=False))
print()

print("=" * 60)
print("All done! ✅")
print("=" * 60)
