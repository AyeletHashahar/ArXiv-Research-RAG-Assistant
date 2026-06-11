import os
import pandas as pd

INPUT_PATH = "data/arxiv-summarization.csv"
OUTPUT_PATH = "data/chunks.csv"

CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500
BATCH_SIZE = 1000


def chunk_article(article):
    chunks = []

    if not isinstance(article, str) or article.strip() == "":
        return chunks

    start_idx = 0

    while start_idx < len(article):
        end_idx = min(start_idx + CHUNK_SIZE, len(article))
        chunks.append((start_idx, end_idx, article[start_idx:end_idx]))

        if end_idx == len(article):
            break

        start_idx = end_idx - CHUNK_OVERLAP

    return chunks


def append_batch_to_csv(batch):
    if not batch:
        return

    batch_df = pd.DataFrame(batch)
    file_exists = os.path.exists(OUTPUT_PATH)

    batch_df.to_csv(
        OUTPUT_PATH,
        mode="a",
        header=not file_exists,
        index=False
    )


def create_chunks_in_batches(df):
    if os.path.exists(OUTPUT_PATH):
        os.remove(OUTPUT_PATH)

    batch = []
    total_chunks = 0
    processed_papers = 0

    for i, row in df.iterrows():
        article = row["article"]
        abstract = row["abstract"]

        article_chunks = chunk_article(article)

        if not article_chunks:
            continue

        processed_papers += 1

        for chunk_index, (start_idx, end_idx, chunk_text) in enumerate(article_chunks):
            batch.append({
                "paper_id": row.get("id", i),
                "chunk_id": f"{row.get('id', i)}_{chunk_index}",
                "chunk_index": chunk_index,
                "start_idx": start_idx,
                "end_idx": end_idx,
                "text": chunk_text,
                "abstract": abstract,
            })

            total_chunks += 1

            if len(batch) >= BATCH_SIZE:
                append_batch_to_csv(batch)
                batch = []

    append_batch_to_csv(batch)

    avg_chunks = total_chunks / processed_papers if processed_papers > 0 else 0

    print(f"Processed papers: {processed_papers}")
    print(f"Created chunks: {total_chunks}")
    print(f"Average chunks per paper: {avg_chunks:.2f}")
    print(f"Saved chunks to {OUTPUT_PATH}")


def load_data():
    return pd.read_csv(INPUT_PATH)


def main():
    df = load_data()
    create_chunks_in_batches(df)


if __name__ == "__main__":
    main()