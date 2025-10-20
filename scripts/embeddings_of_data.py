#!/usr/bin/env python3
"""Generate OpenAI embeddings for long-form survey responses."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable, List

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

# Columns mapped from open-ended prompts in data/raw/survey_questions.txt
TEXT_COLUMNS = [
    "Q3_artist_that_pulled_you_in",
    "Q5_Music_formal_change_impact",
    "Q16_Music_guilty_pleasure_text_OE",
    "Q18_Life_theme_song",
    "Q19_Lyric_that_stuck_with_you",
]

DEFAULT_INPUT_CSV = Path("data/raw/music_survey_data.csv")
DEFAULT_OUTPUT_CSV = Path("data/processed/music_survey_data_with_embeddings.csv")
DEFAULT_MODEL = "text-embedding-3-large"
DEFAULT_BATCH_SIZE = 64
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY_SECONDS = 5


def chunked(iterable: List[str], size: int) -> Iterable[List[str]]:
    """Yield successive batches from a list."""
    for start in range(0, len(iterable), size):
        yield iterable[start : start + size]


def get_embeddings_with_retry(
    client: OpenAI,
    texts: List[str],
    model: str,
    max_retries: int,
    delay_seconds: int,
) -> List[List[float]]:
    """Call the embeddings API with basic retry logic."""
    for attempt in range(max_retries):
        try:
            processed_texts = [t if t.strip() else " " for t in texts]
            response = client.embeddings.create(input=processed_texts, model=model)
            return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001 - surface full exception
            if attempt >= max_retries - 1:
                raise
            print(
                f"Embedding request failed (attempt {attempt + 1}/{max_retries}): {exc}. "
                f"Retrying in {delay_seconds} seconds..."
            )
            time.sleep(delay_seconds)
    return [[] for _ in texts]


def stringify_embedding(value: List[float] | None) -> str:
    """Convert embedding vectors to JSON strings for CSV output."""
    if value is None:
        return "[]"
    if not value:
        return "[]"
    return json.dumps(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to the survey CSV file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="Where to store the CSV with embeddings.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="OpenAI embedding model to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Number of rows to embed per request.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=DEFAULT_MAX_RETRIES,
        help="Maximum API retry count per request.",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=DEFAULT_RETRY_DELAY_SECONDS,
        help="Seconds to wait between retries.",
    )
    parser.add_argument(
        "--columns",
        nargs="*",
        default=TEXT_COLUMNS,
        help="Optional override for text columns to embed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not found in environment variables.")

    client = OpenAI(api_key=api_key)

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    df = pd.read_csv(args.input)
    print(f"Loaded data: {df.shape[0]} rows, {df.shape[1]} columns")

    columns_to_embed = [col for col in args.columns if col in df.columns]
    missing_columns = [col for col in args.columns if col not in df.columns]
    if missing_columns:
        print(f"Warning: missing columns skipped: {', '.join(missing_columns)}")

    if not columns_to_embed:
        raise ValueError("No valid columns to embed. Check column names.")

    for column in columns_to_embed:
        print(f"Embedding column '{column}'...")
        texts = df[column].fillna(" ").astype(str).tolist()
        if not any(text.strip() for text in texts):
            print(f"Column '{column}' is empty. Filling with empty embeddings.")
            df[f"{column}_embedding"] = [[] for _ in texts]
            continue

        embeddings: List[List[float]] = []
        for batch in chunked(texts, args.batch_size):
            vectors = get_embeddings_with_retry(
                client,
                batch,
                model=args.model,
                max_retries=args.max_retries,
                delay_seconds=args.retry_delay,
            )
            embeddings.extend(vectors)

        if len(embeddings) != len(df):
            raise RuntimeError(
                f"Embedding count mismatch for column '{column}': "
                f"expected {len(df)}, got {len(embeddings)}"
            )

        df[f"{column}_embedding"] = embeddings
        print(
            f"Finished '{column}'. Example dimension: "
            f"{len(embeddings[0]) if embeddings and embeddings[0] else 0}"
        )

    df_with_strings = df.copy()
    for column in columns_to_embed:
        embed_col = f"{column}_embedding"
        df_with_strings[embed_col] = df_with_strings[embed_col].apply(
            stringify_embedding
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df_with_strings.to_csv(args.output, index=False)
    print(f"Embeddings saved to {args.output}")


if __name__ == "__main__":
    main()
