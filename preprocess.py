"""
preprocess.py
=============
Loads the labeled restaurant review data, performs EDA (sentiment distribution
plot), splits into train / val / test sets with stratification, drops
unnecessary columns, and applies NLP preprocessing (cleaning, tokenisation,
lowercasing, lemmatisation) using NLTK.

The public API consumed by other scripts:
    load_and_preprocess()  →  X_train, X_val, X_test, y_train, y_val, y_test
"""

import re
import os
import warnings

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ── NLTK resources ──────────────────────────────────────────────────────────
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

# ── Constants ───────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "restoran_jakarta_labeled.csv")
RANDOM_STATE = 42
TEST_SIZE = 0.2   # 20 % for test
VAL_SIZE = 0.125  # 12.5 % of remaining 80 % → 10 % of total  (70/10/20 split)
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

# ────────────────────────────────────────────────────────────────────────────
# 1. Load Data
# ────────────────────────────────────────────────────────────────────────────
def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the labelled CSV into a pandas DataFrame."""
    df = pd.read_csv(path)
    print(f"[INFO] Loaded {len(df)} rows from {path}")
    print(f"[INFO] Columns: {list(df.columns)}")
    print(f"[INFO] Shape : {df.shape}")
    print()
    print(df.head())
    print()
    print(df.info())
    print()
    return df

# ────────────────────────────────────────────────────────────────────────────
# 2. EDA – Sentiment Distribution
# ────────────────────────────────────────────────────────────────────────────
def plot_sentiment_distribution(df: pd.DataFrame) -> None:
    """Plot and save a bar chart of the sentiment class distribution."""
    os.makedirs(PLOT_DIR, exist_ok=True)

    sentiment_counts = df["sentiment"].value_counts()
    print("[INFO] Sentiment distribution:")
    print(sentiment_counts)
    print()

    # ── Plot ──
    palette = {"positive": "#2ecc71", "neutral": "#f39c12", "negative": "#e74c3c"}
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.countplot(
        data=df,
        x="sentiment",
        order=["positive", "neutral", "negative"],
        palette=palette,
        ax=ax,
    )

    # Annotate bars with counts and percentages
    total = len(df)
    for p in ax.patches:
        count = int(p.get_height())
        pct = count / total * 100
        ax.annotate(
            f"{count}\n({pct:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, p.get_height()),
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_title("Sentiment Distribution", fontsize=16, fontweight="bold")
    ax.set_xlabel("Sentiment", fontsize=13)
    ax.set_ylabel("Count", fontsize=13)
    plt.tight_layout()

    save_path = os.path.join(PLOT_DIR, "sentiment_distribution.png")
    fig.savefig(save_path, dpi=150)
    plt.show()
    print(f"[INFO] Plot saved to {save_path}")
    print()

# ────────────────────────────────────────────────────────────────────────────
# 3. Drop Unnecessary Columns
# ────────────────────────────────────────────────────────────────────────────
def drop_unnecessary_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only 'rating', 'reviews' (description text), and 'sentiment'."""
    keep_cols = ["rating", "reviews", "sentiment"]
    df = df[keep_cols].copy()
    print(f"[INFO] Kept columns: {list(df.columns)}")
    print(df.head())
    print()
    return df

# ────────────────────────────────────────────────────────────────────────────
# 4. Train / Validation / Test Split  (stratified)
# ────────────────────────────────────────────────────────────────────────────
def split_data(df: pd.DataFrame):
    """
    Stratified split → 70 % train · 10 % validation · 20 % test.
    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    X = df.drop(columns=["sentiment"])
    y = df["sentiment"]

    # First split: 80 % train+val, 20 % test
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Second split: from the 80 %, take 12.5 % as validation → 10 % of total
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=VAL_SIZE,
        stratify=y_train_val,
        random_state=RANDOM_STATE,
    )

    print(f"[INFO] Train size : {len(X_train)}  ({len(X_train)/len(df)*100:.1f}%)")
    print(f"[INFO] Val   size : {len(X_val)}   ({len(X_val)/len(df)*100:.1f}%)")
    print(f"[INFO] Test  size : {len(X_test)}  ({len(X_test)/len(df)*100:.1f}%)")
    print()

    return X_train, X_val, X_test, y_train, y_val, y_test

# ────────────────────────────────────────────────────────────────────────────
# 5. Text Preprocessing  (NLTK)
# ────────────────────────────────────────────────────────────────────────────
_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("indonesian"))  # Indonesian stop words

def clean_text(text: str) -> str:
    """
    Pipeline: lowercase → remove URLs → remove mentions/hashtags →
    remove non-alpha characters → tokenise → remove stop words → lemmatise →
    rejoin.
    """
    if not isinstance(text, str) or text.strip() == "":
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", "", text)

    # Remove mentions & hashtags
    text = re.sub(r"@\w+|#\w+", "", text)

    # Remove emojis / non‑ASCII decorations
    text = re.sub(
        r"[^\w\s]", "", text, flags=re.UNICODE
    )

    # Remove digits
    text = re.sub(r"\d+", "", text)

    # Tokenise
    tokens = word_tokenize(text)

    # Remove stop words & lemmatise
    tokens = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in _stop_words and len(tok) > 1
    ]

    return " ".join(tokens)


def preprocess_text_column(df: pd.DataFrame, col: str = "reviews") -> pd.DataFrame:
    """Apply *clean_text* to a text column in‑place and return the DataFrame."""
    df = df.copy()
    print(f"[INFO] Preprocessing column '{col}' …")
    df[col] = df[col].apply(clean_text)
    print("[INFO] Done.\n")
    print(df.head())
    print()
    return df

# ────────────────────────────────────────────────────────────────────────────
# 6. Public API – single entry‑point for other modules
# ────────────────────────────────────────────────────────────────────────────
def load_and_preprocess():
    """
    Full pipeline:
        load → EDA plot → drop columns → split → preprocess text.
    Returns
    -------
    X_train, X_val, X_test, y_train, y_val, y_test
        where each X is a DataFrame with columns ['rating', 'reviews'].
    """
    df = load_data()
    plot_sentiment_distribution(df)
    df = drop_unnecessary_columns(df)

    # Drop rows where 'reviews' is NaN / empty
    df = df.dropna(subset=["reviews"])
    df = df[df["reviews"].str.strip() != ""]
    df = df.reset_index(drop=True)
    print(f"[INFO] After dropping empty reviews: {len(df)} rows remain.\n")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)

    # Preprocess text for each split independently
    X_train = preprocess_text_column(X_train)
    X_val   = preprocess_text_column(X_val)
    X_test  = preprocess_text_column(X_test)

    return X_train, X_val, X_test, y_train, y_val, y_test


# ────────────────────────────────────────────────────────────────────────────
# Run when executed directly (for quick sanity check)
# ────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess()
    print("=" * 60)
    print("Preprocessing complete.")
    print(f"  Train  → {X_train.shape}")
    print(f"  Val    → {X_val.shape}")
    print(f"  Test   → {X_test.shape}")
    print("=" * 60)
