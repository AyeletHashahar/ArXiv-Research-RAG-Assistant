"""Download the source corpus and load it as a DataFrame.

NOTE ON DOMAIN: despite the project's working name, the corpus wired up here
is `ccdv/arxiv-summarization` (general computer-science papers), not a
medical dataset. The pipeline is domain-agnostic — swapping in a real
medical corpus (e.g. PubMedQA) only requires changing this module. See the
README "Future Improvements" section.
"""

import logging

import pandas as pd
from datasets import load_dataset

from config import DATASET_NAME, DATASET_SPLIT, DATASET_SAMPLE_SIZE, RAW_DATA_PATH

logger = logging.getLogger(__name__)


def download_dataset(sample_size=DATASET_SAMPLE_SIZE):
    """Stream `sample_size` rows from the source dataset and save as CSV."""
    RAW_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Streaming %d rows from %s", sample_size, DATASET_NAME)
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=True)

    rows = []
    for i, sample in enumerate(dataset.take(sample_size)):
        rows.append({"id": i, "article": sample["article"], "abstract": sample["abstract"]})

    if not rows:
        raise ValueError(f"No rows were streamed from dataset '{DATASET_NAME}'")

    df = pd.DataFrame(rows)
    df.to_csv(RAW_DATA_PATH, index=False)
    logger.info("Saved %d rows to %s", len(df), RAW_DATA_PATH)
    return df


def load_data():
    """Load the previously downloaded raw dataset."""
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"{RAW_DATA_PATH} not found. Run `python scripts/load_data.py` first."
        )
    return pd.read_csv(RAW_DATA_PATH)
