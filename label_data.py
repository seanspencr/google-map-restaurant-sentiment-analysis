import os
import re
import time
import logging
import pandas as pd

from adala.agents import Agent
from adala.environments import StaticEnvironment
from adala.skills import ClassificationSkill
from adala.runtimes import OpenAIChatRuntime
from rich import print

# Suppress the "Failed to get cost for model" warning from LiteLLM.
# This fires when a model isn't in LiteLLM's pricing DB — it is harmless.
logging.getLogger("LiteLLM").setLevel(logging.ERROR)
logging.getLogger("LiteLLM Router").setLevel(logging.ERROR)

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_CSV  = "restoran_jakarta_2000_split.csv"
OUTPUT_CSV = "restoran_jakarta_labeled.csv"

FREE_MODEL = "inclusionai/ring-2.6-1t:free"

# How many reviews to send in ONE API call.
# Adala is row-by-row (1 call per row). We bypass it for inference and
# instead pack BATCH_SIZE reviews into a single prompt → far fewer calls.
#   50 rows × 1 call/row  = 50 calls  (Adala default)
#  100 rows × 1 call/100  =  1 call   (our approach)
# Free tier limits apply per-call, so bigger = fewer hits.
BATCH_SIZE = 100   # tune up/down based on your model's context window

# Seconds to wait between batch calls to avoid rate-limit errors.
CALL_DELAY = 2
# ──────────────────────────────────────────────────────────────────────────────


def build_runtime() -> OpenAIChatRuntime:
    """Return a configured OpenAIChatRuntime pointing at OpenRouter."""
    return OpenAIChatRuntime(
        base_url="https://openrouter.ai/api/v1",
        model=FREE_MODEL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        provider="Custom"
    )


def derive_training_set(df: pd.DataFrame, n_per_class: int = 5) -> pd.DataFrame:
    """
    Build a labelled training DataFrame by sampling from the CSV itself,
    using the `rating` column as a proxy for sentiment:
      - rating 4–5  → Positive
      - rating 3    → Neutral
      - rating 1–2  → Negative

    Only rows that have non-empty review text are considered.
    """
    has_text = df["reviews"].notna() & (df["reviews"].str.strip() != "")
    has_rating = df["rating"].notna()
    base = df[has_text & has_rating].copy()

    def _map_rating(r: float) -> str:
        if r >= 4:
            return "Positive"
        elif r == 3:
            return "Neutral"
        else:
            return "Negative"

    base["sentiment"] = base["rating"].apply(_map_rating)

    frames = []
    for label in ["Positive", "Neutral", "Negative"]:
        subset = base[base["sentiment"] == label]
        sampled = subset.sample(n=min(n_per_class, len(subset)), random_state=42)
        frames.append(sampled[["reviews", "sentiment"]])

    train_df = pd.concat(frames, ignore_index=True)
    return train_df


def build_agent(train_df: pd.DataFrame) -> Agent:
    """
    Build an Adala Agent using the provided training DataFrame.
    The training set should have columns ['reviews', 'sentiment'].
    """

    runtime = build_runtime()

    agent = Agent(
        environment=StaticEnvironment(df=train_df),
        skills=ClassificationSkill(
            name="sentiment",
            instructions=(
                "You are a sentiment analysis expert for Indonesian restaurant reviews. "
                "Classify each review as exactly one of: Positive, Negative, or Neutral. "
                "Positive: the reviewer is satisfied, happy, or praises the restaurant. "
                "Negative: the reviewer is dissatisfied, complains, or criticises. "
                "Neutral: the review is mixed, indifferent, or lacks a clear sentiment. "
                "The review may be written in Indonesian (Bahasa Indonesia) or English."
            ),
            labels=["Positive", "Negative", "Neutral"],
            input_template="Review: {reviews}",
            output_template="Sentiment: {sentiment}",
        ),
        runtimes={"openrouter": runtime},
        default_runtime="openrouter",
        teacher_runtimes={"default": build_runtime()},
    )
    return agent


# ── Direct batched labeler (bypasses Adala's row-by-row loop) ────────────────
SYSTEM_PROMPT = (
    "You are a sentiment analysis expert for Indonesian and English restaurant reviews. "
    "Classify each review as EXACTLY one of: Positive, Negative, or Neutral.\n"
    "Positive: reviewer is satisfied, happy, or praises the restaurant.\n"
    "Negative: reviewer is dissatisfied, complains, or criticises.\n"
    "Neutral : review is mixed, indifferent, or lacks clear sentiment."
)


def _parse_labels(response_text: str, n: int) -> list[str]:
    """
    Parse a numbered response like:
      1. Positive
      2. Negative
    Returns a list of n labels; falls back to 'Neutral' for unparseable lines.
    """
    valid = {"Positive", "Negative", "Neutral"}
    labels: list[str] = []
    for line in response_text.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        # strip leading number and punctuation: "1. Positive" → "Positive"
        cleaned = re.sub(r"^\d+[.)\s]+", "", line).strip()
        # pick first matching label word anywhere in the line
        found = next((v for v in valid if v.lower() in cleaned.lower()), None)
        if found:
            labels.append(found)
    # pad or truncate to exactly n
    while len(labels) < n:
        labels.append("Neutral")
    return labels[:n]


def label_in_batches(df: pd.DataFrame) -> pd.Series:
    """
    Label all reviews in df using ONE API call per BATCH_SIZE rows.

    Adala processes rows one-by-one (1 call per row). This function packs
    BATCH_SIZE reviews into a single prompt, returning all labels at once —
    drastically reducing API calls and rate-limit pressure.
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
    )

    results: dict[int, str] = {}
    total = len(df)
    n_batches = (total + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num, start in enumerate(range(0, total, BATCH_SIZE), 1):
        batch = df.iloc[start : start + BATCH_SIZE]
        reviews = batch["reviews"].tolist()
        n = len(reviews)

        print(
            f"  Batch {batch_num}/{n_batches}: rows {start + 1}–{start + n} / {total} "
            f"({n} reviews in 1 API call) …"
        )

        # Build a numbered list of reviews in a single user message
        numbered = "\n".join(f"{i + 1}. {r}" for i, r in enumerate(reviews))
        user_msg = (
            f"Label each review below as Positive, Negative, or Neutral.\n"
            f"Reply ONLY with a numbered list — one label per line, same order.\n\n"
            f"{numbered}"
        )

        try:
            response = client.chat.completions.create(
                model=FREE_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_msg},
                ],
                # cap output tokens: each label is ≤ 3 words + number ≈ 10 tokens
                max_tokens=n * 12,
                temperature=0,
            )
            raw = response.choices[0].message.content or ""
            labels = _parse_labels(raw, n)
        except Exception as exc:
            print(f"  [yellow]Batch {batch_num} failed: {exc} — defaulting to Neutral[/yellow]")
            labels = ["Neutral"] * n

        for idx, label in zip(batch.index, labels):
            results[idx] = label

        # polite delay to stay within rate limits
        if start + BATCH_SIZE < total:
            time.sleep(CALL_DELAY)

    return pd.Series(results, name="sentiment")


def main() -> None:
    print(f"\n[bold cyan]Loading[/bold cyan] {INPUT_CSV} …")
    df = pd.read_csv(INPUT_CSV)

    print(f"[bold]Dataset shape:[/bold] {df.shape}")
    print(f"[bold]Columns:[/bold] {list(df.columns)}\n")

    # ── Separate rows with and without review text ────────────────────────────
    has_review  = df["reviews"].notna() & (df["reviews"].str.strip() != "")
    df_with     = df[has_review].copy()
    df_without  = df[~has_review].copy()

    print(
        f"Rows with reviews   : [green]{len(df_with)}[/green]\n"
        f"Rows without reviews: [yellow]{len(df_without)}[/yellow] → labelled as 'Neutral'\n"
    )

    # ── Derive training set from the data itself ─────────────────────────────
    print("[bold cyan]Deriving training set from CSV (using rating column) …[/bold cyan]")
    train_df = derive_training_set(df, n_per_class=5)
    print(f"Training examples per class (sampled from CSV):")
    print(train_df["sentiment"].value_counts().to_string())
    print()

    # ── Build agent ───────────────────────────────────────────────────────────
    print("[bold cyan]Building Adala agent …[/bold cyan]")
    agent = build_agent(train_df)

    # Optional: run a short learning pass (3 iterations) to refine instructions.
    # Comment this out if you want to skip and just run inference directly.
    print("\n[bold cyan]Running learning phase (3 iterations) …[/bold cyan]")
    try:
        agent.learn(learning_iterations=3, accuracy_threshold=0.95)
    except Exception as exc:
        print(f"[yellow]Learning phase skipped due to error: {exc}[/yellow]")

    # ── Inference (direct batched calls — not Adala row-by-row) ──────────────
    print(f"\n[bold cyan]Labelling reviews … ({len(df_with)} rows, batch size {BATCH_SIZE})[/bold cyan]")
    print(f"[dim]≈ {(len(df_with) + BATCH_SIZE - 1) // BATCH_SIZE} API calls total "
          f"(vs {len(df_with)} with Adala default)[/dim]\n")
    predicted = label_in_batches(df_with)

    # ── Merge results back into the full dataframe ────────────────────────────
    df.loc[has_review,  "sentiment"] = predicted
    df.loc[~has_review, "sentiment"] = "Neutral"  # no text → neutral by default

    # ── Save ──────────────────────────────────────────────────────────────────
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[bold green]✓ Saved labelled data to:[/bold green] {OUTPUT_CSV}")

    # ── Quick summary ─────────────────────────────────────────────────────────
    print("\n[bold]Sentiment distribution:[/bold]")
    print(df["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    main()
