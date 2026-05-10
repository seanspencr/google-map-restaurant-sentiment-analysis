"""
train_lstm.py
=============
Trains a BiLSTM sentiment classifier on restaurant reviews.

Feature extraction : Keras Tokenizer → padded integer sequences
                     (GloVe 100d is used as the Embedding initialiser,
                      which requires sequential word‑index input – not TF‑IDF)

Architecture (as specified):
    Input
      → Embedding (GloVe 100d, non‑trainable)
      → Bidirectional LSTM (64 units, return_sequences=True)
      → Dropout (0.4)
      → LSTM (32 units)
      → Dropout (0.4)
      → Dense (32, relu)
      → Dense (3, softmax)          # output layer – 3 classes

Evaluation : Accuracy, Precision, Recall, F1
Power      : CodeCarbon for a single classification task
"""

import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding,
    Bidirectional,
    LSTM,
    Dropout,
    Dense,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

from codecarbon import EmissionsTracker

# Re‑use the shared preprocessing pipeline
from preprocess import load_and_preprocess

warnings.filterwarnings("ignore")

# ── Constants ───────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(__file__)
PLOT_DIR      = os.path.join(BASE_DIR, "plots")
GLOVE_DIR     = os.path.join(BASE_DIR, "glove")

# HuggingFace repo hosting GloVe vectors (faster than Stanford servers)
GLOVE_HF_REPO = "stanfordnlp/glove"            # repo_id on HuggingFace
GLOVE_HF_FILE = "glove.6B.zip"                 # the zip hosted in the repo
GLOVE_TXT     = "glove.6B.100d.txt"             # file we need from inside the zip
GLOVE_LOCAL   = os.path.join(GLOVE_DIR, GLOVE_TXT)

MAX_WORDS     = 10_000     # vocabulary cap
MAX_LEN       = 200        # max sequence length (tokens)
EMBEDDING_DIM = 100        # GloVe dimension
BATCH_SIZE    = 32
EPOCHS        = 50

os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(GLOVE_DIR, exist_ok=True)


# ────────────────────────────────────────────────────────────────────────────
# Helper – Download GloVe embeddings from HuggingFace Hub
# ────────────────────────────────────────────────────────────────────────────
def _download_glove() -> str:
    """
    Download glove.6B.zip from HuggingFace Hub, extract the 100d txt file.
    Returns the local path to glove.6B.100d.txt.
    """
    import zipfile
    from huggingface_hub import hf_hub_download

    # If already extracted, skip
    if os.path.isfile(GLOVE_LOCAL):
        print(f"[INFO] GloVe file already exists: {GLOVE_LOCAL}")
        return GLOVE_LOCAL

    print(f"[INFO] Downloading GloVe zip from HuggingFace: {GLOVE_HF_REPO}/{GLOVE_HF_FILE}")
    zip_path = hf_hub_download(
        repo_id=GLOVE_HF_REPO,
        filename=GLOVE_HF_FILE,
    )
    print(f"[INFO] Zip downloaded to: {zip_path}")

    # Extract only the 100d file
    print(f"[INFO] Extracting {GLOVE_TXT} …")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extract(GLOVE_TXT, GLOVE_DIR)

    print(f"[INFO] GloVe file ready at: {GLOVE_LOCAL}")
    return GLOVE_LOCAL


def load_glove_index(path: str) -> dict:
    """Load GloVe vectors into a {word: np.array} dictionary."""
    print(f"[INFO] Loading GloVe vectors from {path} …")
    embeddings_index: dict[str, np.ndarray] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word = parts[0]
            coefs = np.asarray(parts[1:], dtype="float32")
            embeddings_index[word] = coefs
    print(f"[INFO] Loaded {len(embeddings_index):,} word vectors.")
    return embeddings_index


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
# STEP 2 – Feature Extraction (Tokeniser → Padded Sequences)
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 – Feature Extraction (Tokeniser + GloVe 100d Embedding)")
print("=" * 60)

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train["reviews"])

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train["reviews"]),
                            maxlen=MAX_LEN, padding="post", truncating="post")
X_val_seq   = pad_sequences(tokenizer.texts_to_sequences(X_val["reviews"]),
                            maxlen=MAX_LEN, padding="post", truncating="post")
X_test_seq  = pad_sequences(tokenizer.texts_to_sequences(X_test["reviews"]),
                            maxlen=MAX_LEN, padding="post", truncating="post")

word_index = tokenizer.word_index
vocab_size = min(MAX_WORDS, len(word_index) + 1)

print(f"[INFO] Unique tokens  : {len(word_index):,}")
print(f"[INFO] Vocabulary cap : {MAX_WORDS}")
print(f"[INFO] Sequence length: {MAX_LEN}")
print(f"[INFO] Train shape    : {X_train_seq.shape}")
print(f"[INFO] Val   shape    : {X_val_seq.shape}")
print(f"[INFO] Test  shape    : {X_test_seq.shape}")
print()

# ── Label encoding (string → integer → one‑hot) ──
label_encoder = LabelEncoder()
label_encoder.fit(["negative", "neutral", "positive"])   # fixed order

y_train_enc = to_categorical(label_encoder.transform(y_train), num_classes=3)
y_val_enc   = to_categorical(label_encoder.transform(y_val),   num_classes=3)
y_test_enc  = to_categorical(label_encoder.transform(y_test),  num_classes=3)

print(f"[INFO] Label classes  : {list(label_encoder.classes_)}")
print(f"[INFO] y_train shape  : {y_train_enc.shape}")
print()

# ────────────────────────────────────────────────────────────────────────────
# STEP 3 – Build GloVe Embedding Matrix
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 – Building GloVe Embedding Matrix")
print("=" * 60)

glove_path = _download_glove()
glove_index = load_glove_index(glove_path)

embedding_matrix = np.zeros((vocab_size, EMBEDDING_DIM))
hits = 0
for word, idx in word_index.items():
    if idx >= vocab_size:
        continue
    vec = glove_index.get(word)
    if vec is not None:
        embedding_matrix[idx] = vec
        hits += 1

print(f"[INFO] Embedding matrix shape : {embedding_matrix.shape}")
print(f"[INFO] Vocab words with GloVe : {hits}/{vocab_size-1} "
      f"({hits / (vocab_size - 1) * 100:.1f}%)")
print()

# ────────────────────────────────────────────────────────────────────────────
# STEP 4 – Build & Train BiLSTM Model
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4 – Model Architecture & Training")
print("=" * 60)

model = Sequential([
    # Embedding layer – initialised with GloVe, frozen
    Embedding(
        input_dim=vocab_size,
        output_dim=EMBEDDING_DIM,
        weights=[embedding_matrix],
        input_length=MAX_LEN,
        trainable=False,
        name="glove_embedding",
    ),
    # BiLSTM 64 (return sequences for next LSTM)
    Bidirectional(LSTM(64, return_sequences=True), name="bilstm_64"),
    Dropout(0.4, name="dropout_1"),
    # LSTM 32
    LSTM(32, name="lstm_32"),
    Dropout(0.4, name="dropout_2"),
    # Dense 32
    Dense(32, activation="relu", name="dense_32"),
    # Output layer – 3 classes
    Dense(3, activation="softmax", name="output"),
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()
print()

# ── Callbacks ──
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
    verbose=1,
)

# ── Training ──
history = model.fit(
    X_train_seq, y_train_enc,
    validation_data=(X_val_seq, y_val_enc),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop],
    verbose=1,
)

# ── Training curves ──
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history["loss"],     label="Train Loss")
axes[0].plot(history.history["val_loss"], label="Val Loss")
axes[0].set_title("Loss", fontsize=14, fontweight="bold")
axes[0].set_xlabel("Epoch")
axes[0].set_ylabel("Loss")
axes[0].legend()

axes[1].plot(history.history["accuracy"],     label="Train Accuracy")
axes[1].plot(history.history["val_accuracy"], label="Val Accuracy")
axes[1].set_title("Accuracy", fontsize=14, fontweight="bold")
axes[1].set_xlabel("Epoch")
axes[1].set_ylabel("Accuracy")
axes[1].legend()

plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "lstm_training_curves.png"), dpi=150)
plt.show()
print(f"[INFO] Training curves saved to {PLOT_DIR}/lstm_training_curves.png\n")

# ────────────────────────────────────────────────────────────────────────────
# STEP 5 – Evaluation
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5 – Evaluation on Test Set")
print("=" * 60)

y_pred_proba = model.predict(X_test_seq, verbose=0)
y_pred_idx   = np.argmax(y_pred_proba, axis=1)
y_true_idx   = np.argmax(y_test_enc,   axis=1)

y_pred_labels = label_encoder.inverse_transform(y_pred_idx)
y_true_labels = label_encoder.inverse_transform(y_true_idx)

acc  = accuracy_score(y_true_labels, y_pred_labels)
prec = precision_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
rec  = recall_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)
f1   = f1_score(y_true_labels, y_pred_labels, average="weighted", zero_division=0)

print(f"\n{'─' * 40}")
print(f"  BiLSTM (GloVe 100d)")
print(f"{'─' * 40}")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")
print()
print(classification_report(y_true_labels, y_pred_labels, zero_division=0))

# ── Confusion Matrix ──
cm = confusion_matrix(y_true_labels, y_pred_labels, labels=["positive", "neutral", "negative"])
disp = ConfusionMatrixDisplay(cm, display_labels=["positive", "neutral", "negative"])
fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap="Blues")
ax.set_title("Confusion Matrix – BiLSTM", fontsize=14, fontweight="bold")
plt.tight_layout()
fig.savefig(os.path.join(PLOT_DIR, "cm_bilstm.png"), dpi=150)
plt.show()
print(f"[INFO] Confusion matrix saved to {PLOT_DIR}/cm_bilstm.png\n")

# ── Summary table ──
results_df = pd.DataFrame([{
    "Model":     "BiLSTM (GloVe 100d)",
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
# STEP 6 – Power Consumption (CodeCarbon) – single classification task
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 6 – Power Consumption (CodeCarbon)")
print("=" * 60)

# Single sample from test set
single_sample = X_test_seq[0:1]  # keep 2D shape (1, MAX_LEN)

tracker = EmissionsTracker(
    project_name="single_predict_BiLSTM",
    measure_power_secs=0.1,
    log_level="error",
    save_to_file=False,
)

tracker.start()
_ = model.predict(single_sample, verbose=0)
emissions = tracker.stop()  # kg CO₂

energy_kwh = tracker.final_emissions_data.energy_consumed  # kWh

print(f"\n[BiLSTM]  Energy: {energy_kwh:.10f} kWh  |  CO₂: {emissions:.10f} kg")

energy_df = pd.DataFrame([{
    "Model":              "BiLSTM (GloVe 100d)",
    "Energy (kWh)":       f"{energy_kwh:.10f}",
    "CO₂ Emissions (kg)": f"{emissions:.10f}",
}])
print()
print(energy_df.to_string(index=False))
print()

print("=" * 60)
print("All done! ✅")
print("=" * 60)
