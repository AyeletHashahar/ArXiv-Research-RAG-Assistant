"""Split long documents into overlapping, fixed-size text chunks."""

import logging

import pandas as pd

from config import CHUNK_OVERLAP, CHUNK_SIZE, CHUNK_WRITE_BATCH_SIZE, CHUNKS_PATH

logger = logging.getLogger(__name__)


def chunk_article(article, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    """Slide a `chunk_size` window over `article` with `overlap` characters
    of overlap between consecutive chunks. Returns [(start, end, text), ...].
    """
    if not isinstance(article, str) or article.strip() == "":
        return []

    chunks = []
    start_idx = 0

    while start_idx < len(article):
        end_idx = min(start_idx + chunk_size, len(article))
        chunks.append((start_idx, end_idx, article[start_idx:end_idx]))

        if end_idx == len(article):
            break

        start_idx = end_idx - overlap

    return chunks


def _append_batch_to_csv(batch, output_path):
    if not batch:
        return

    batch_df = pd.DataFrame(batch)
    file_exists = output_path.exists()
    batch_df.to_csv(output_path, mode="a", header=not file_exists, index=False)


def create_chunks_in_batches(df, output_path=CHUNKS_PATH, batch_size=CHUNK_WRITE_BATCH_SIZE):
    """Chunk every article in `df` and stream the results to `output_path`
    in batches, so memory use stays flat regardless of corpus size.
    """
    if df.empty:
        raise ValueError("Input DataFrame has no rows to chunk")

    if output_path.exists():
        output_path.unlink()

    batch = []
    total_chunks = 0
    processed_papers = 0
    seen_chunk_ids = set()

    for i, row in df.iterrows():
        article_chunks = chunk_article(row["article"])
        if not article_chunks:
            continue

        processed_papers += 1

        for chunk_index, (start_idx, end_idx, chunk_text) in enumerate(article_chunks):
            paper_id = row.get("id", i)
            chunk_id = f"{paper_id}_{chunk_index}"

            if chunk_id in seen_chunk_ids:
                logger.warning("Skipping duplicate chunk_id: %s", chunk_id)
                continue
            seen_chunk_ids.add(chunk_id)

            batch.append(
                {
                    "paper_id": paper_id,
                    "chunk_id": chunk_id,
                    "chunk_index": chunk_index,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                    "text": chunk_text,
                    "abstract": row.get("abstract"),
                }
            )
            total_chunks += 1

            if len(batch) >= batch_size:
                _append_batch_to_csv(batch, output_path)
                batch = []

    _append_batch_to_csv(batch, output_path)

    if total_chunks == 0:
        raise ValueError("No chunks were produced from the input DataFrame")

    avg_chunks = total_chunks / processed_papers if processed_papers else 0
    logger.info("Processed papers: %d", processed_papers)
    logger.info("Created chunks: %d", total_chunks)
    logger.info("Average chunks per paper: %.2f", avg_chunks)
    logger.info("Saved chunks to %s", output_path)
