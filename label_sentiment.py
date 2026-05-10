"""
Sentiment Labeling Script for Jakarta Restaurant Dataset
=========================================================
Uses Google Gemini Flash (free tier) to label sentiment in batches.
Combines 'deskripsi' and 'reviews' with 'rating' as labeling signals.

Requirements:
    pip install google-generativeai pandas tqdm

Get a free Gemini API key at: https://aistudio.google.com/app/apikey
"""

import os
import time
import random
import textwrap
import pandas as pd
from tqdm import tqdm
from groq import Groq

# ─── CONFIG ───────────────────────────────────────────────────────────────────

INPUT_FILE     = "restoran_jakarta_2000_split.csv"
OUTPUT_FILE    = "restoran_jakarta_labeled.csv"

SAMPLE_SIZE    = 700       # rows to sample (500–1000 recommended)
BATCH_SIZE     = 20        # rows per LLM call — balances tokens vs API calls
RANDOM_SEED    = 42        # reproducibility

GROQ_API_KEY = "your_groq_api_key"
MODEL_NAME   = "llama-3.1-8b-instant"   # free, fast, great for classification

# ─── SETUP ────────────────────────────────────────────────────────────────────


client = Groq(api_key=GROQ_API_KEY)

# ─── STEP 1: LOAD & SAMPLE ────────────────────────────────────────────────────

print(f"Loading {INPUT_FILE} ...")
df_full = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"  Total rows in dataset: {len(df_full):,}")

# Keep only relevant columns
RELEVANT_COLS = ["nama", "area", "rating", "deskripsi", "reviews", "place_id"]
available_cols = [c for c in RELEVANT_COLS if c in df_full.columns]
df = df_full[available_cols].copy()

# Drop rows with no text signal at all
df = df.dropna(subset=["deskripsi", "reviews"], how="all").reset_index(drop=True)

# Random sample
random.seed(RANDOM_SEED)
sample_size = min(SAMPLE_SIZE, len(df))
df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)
print(f"  Sampled {len(df)} rows for labeling.\n")

# ─── STEP 2: BATCH PROMPT BUILDER ─────────────────────────────────────────────

SYSTEM_PROMPT = textwrap.dedent("""
    You are a sentiment analysis expert for Indonesian restaurant reviews.
    You will receive a numbered list of restaurant entries, each with:
      - rating (1–5 stars, may be missing)
      - description (deskripsi)
      - reviews (customer reviews excerpt)

    For EACH entry, classify the overall sentiment as exactly one of:
      positive | neutral | negative

    Rules:
    - "positive"  → generally happy customers, good food/service, rating 4–5
    - "neutral"   → mixed opinions, average experience, rating ~3, or no strong signal
    - "negative"  → complaints, bad food/service, rating 1–2
    - If text is missing or uninformative, use the rating as the main signal.
    - If both are missing, output "neutral".
    - Respond ONLY with a JSON array of objects, one per entry, in the same order.
      Format: [{"id": 1, "sentiment": "positive"}, {"id": 2, "sentiment": "neutral"}, ...]
    - No extra text, no markdown fences, just the raw JSON array.
""").strip()


def build_batch_prompt(batch_df: pd.DataFrame, start_idx: int) -> str:
    """Build a numbered prompt for a batch of rows."""
    lines = []
    for local_i, (_, row) in enumerate(batch_df.iterrows()):
        entry_id = start_idx + local_i + 1
        rating   = row.get("rating", "N/A")
        desc     = str(row.get("deskripsi", "") or "").strip()[:300]   # cap length
        review   = str(row.get("reviews",   "") or "").strip()[:300]
        lines.append(
            f"Entry {entry_id}:\n"
            f"  rating: {rating}\n"
            f"  deskripsi: {desc}\n"
            f"  reviews: {review}"
        )
    return SYSTEM_PROMPT + "\n\n" + "\n\n".join(lines)


# ─── STEP 3: CALL LLM IN BATCHES ──────────────────────────────────────────────

import json
import re

sentiments = []
total_batches = (len(df) + BATCH_SIZE - 1) // BATCH_SIZE

print(f"Labeling {len(df)} rows in {total_batches} batches of up to {BATCH_SIZE} ...\n")

for batch_num in tqdm(range(total_batches), desc="Batches", unit="batch"):
    start = batch_num * BATCH_SIZE
    end   = min(start + BATCH_SIZE, len(df))
    batch = df.iloc[start:end]

    prompt = build_batch_prompt(batch, start)

    # Retry logic (Gemini free tier has rate limits)
    for attempt in range(4):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
            )
            raw = response.choices[0].message.content.strip()

            # Parse JSON — strip accidental markdown fences if present
            raw_clean = re.sub(r"^```[a-z]*\n?|```$", "", raw, flags=re.MULTILINE).strip()
            parsed    = json.loads(raw_clean)

            # Extract labels in order
            batch_labels = [item["sentiment"].lower() for item in sorted(parsed, key=lambda x: x["id"])]

            # Validate labels
            valid = {"positive", "neutral", "negative"}
            batch_labels = [lbl if lbl in valid else "neutral" for lbl in batch_labels]

            # Pad if LLM returned fewer labels than expected
            while len(batch_labels) < len(batch):
                batch_labels.append("neutral")

            sentiments.extend(batch_labels[:len(batch)])
            break  # success

        except (json.JSONDecodeError, KeyError, IndexError) as e:
            print(f"\n  [Batch {batch_num+1}] Parse error (attempt {attempt+1}): {e}")
            if attempt == 3:
                # Fallback: label entire batch as neutral
                sentiments.extend(["neutral"] * len(batch))
        except Exception as e:
            err_str = str(e)
            wait    = 15 * (attempt + 1)
            print(f"\n  [Batch {batch_num+1}] API error (attempt {attempt+1}): {err_str[:80]}")
            if "quota" in err_str.lower() or "429" in err_str:
                wait = 60
            if attempt == 3:
                sentiments.extend(["neutral"] * len(batch))
            else:
                time.sleep(wait)

    # Polite delay between batches (free tier: 15 RPM)
    time.sleep(2)

# ─── STEP 4: ATTACH LABELS & SAVE ─────────────────────────────────────────────

df["sentiment"] = sentiments

# Distribution summary
dist = df["sentiment"].value_counts()
print("\n── Label distribution ──────────────────")
for lbl, cnt in dist.items():
    pct = cnt / len(df) * 100
    print(f"  {lbl:<10} {cnt:>4}  ({pct:.1f}%)")
print("────────────────────────────────────────\n")

df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")
print(f"✓ Labeled CSV saved → {OUTPUT_FILE}")
print(f"  Columns: {list(df.columns)}")