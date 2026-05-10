"""
train_machine_learning.py
=========================
Feature extraction with TF-IDF, training of three ML classifiers
(Naive Bayes, SVM, Logistic Regression), evaluation metrics, and
per-prediction power consumption measurement via CodeCarbon.
"""

import os
import warnings
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
import matplotlib.pyplot as plt
from codecarbon import EmissionsTracker

# Re‑use the preprocessing pipeline
from preprocess import load_and_preprocess

warnings.filterwarnings("ignore")

PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

# ────────────────────────────────────────────────────────────────────────────
# 1. Load & Preprocess Data
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1 – Loading & Preprocessing Data")
print("=" * 60)

X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()

# ────────────────────────────────────────────────────────────────────────────
# 2. Feature Extraction – TF‑IDF
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 2 – TF-IDF Feature Extraction")
print("=" * 60)

tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_tfidf = tfidf.fit_transform(X_train["reviews"])
X_val_tfidf   = tfidf.transform(X_val["reviews"])
X_test_tfidf  = tfidf.transform(X_test["reviews"])

print(f"[INFO] TF-IDF vocabulary size : {len(tfidf.vocabulary_)}")
print(f"[INFO] Train matrix shape     : {X_train_tfidf.shape}")
print(f"[INFO] Val   matrix shape     : {X_val_tfidf.shape}")
print(f"[INFO] Test  matrix shape     : {X_test_tfidf.shape}")
print()

# ────────────────────────────────────────────────────────────────────────────
# 3. Model Training
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 3 – Model Training")
print("=" * 60)

models = {
    "Naive Bayes":          MultinomialNB(),
    "SVM":                  SVC(kernel="linear", C=1.0, random_state=42),
    "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
}

trained_models = {}
for name, model in models.items():
    print(f"[TRAIN] Training {name} …")
    model.fit(X_train_tfidf, y_train)
    trained_models[name] = model
    print(f"[TRAIN] {name} – done.")

print()

# ────────────────────────────────────────────────────────────────────────────
# 4. Evaluation
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 4 – Evaluation on Test Set")
print("=" * 60)

results = []

for name, model in trained_models.items():
    y_pred = model.predict(X_test_tfidf)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results.append({
        "Model":     name,
        "Accuracy":  round(acc, 4),
        "Precision": round(prec, 4),
        "Recall":    round(rec, 4),
        "F1-Score":  round(f1, 4),
    })

    print(f"\n{'─' * 40}")
    print(f"  {name}")
    print(f"{'─' * 40}")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Confusion Matrix ──
    cm = confusion_matrix(y_test, y_pred, labels=["positive", "neutral", "negative"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["positive", "neutral", "negative"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(PLOT_DIR, f"cm_{name.lower().replace(' ', '_')}.png"), dpi=150)
    plt.show()

# ── Summary table ──
print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)
results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))
print()

# ────────────────────────────────────────────────────────────────────────────
# 5. Power Consumption – CodeCarbon  (single classification task)
# ────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 5 – Power Consumption (CodeCarbon)")
print("=" * 60)

# Take ONE sample from the test set for a single classification task
single_sample = X_test_tfidf[0]

energy_results = []

for name, model in trained_models.items():
    tracker = EmissionsTracker(
        project_name=f"single_predict_{name}",
        measure_power_secs=0.1,
        log_level="error",        # suppress verbose logs
        save_to_file=False,
    )

    tracker.start()
    _ = model.predict(single_sample)
    emissions = tracker.stop()    # returns kg CO₂

    energy_kwh = tracker.final_emissions_data.energy_consumed  # kWh

    energy_results.append({
        "Model":                name,
        "Energy (kWh)":         f"{energy_kwh:.10f}",
        "CO₂ Emissions (kg)":   f"{emissions:.10f}",
    })

    print(f"[{name}]  Energy: {energy_kwh:.10f} kWh  |  CO₂: {emissions:.10f} kg")

print()
energy_df = pd.DataFrame(energy_results)
print(energy_df.to_string(index=False))
print()

print("=" * 60)
print("All done! ✅")
print("=" * 60)
