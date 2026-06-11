from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb

CHUNKS_PATH = "data/chunks.csv"
DB_PATH = "chroma_db"
COLLECTION_NAME = "arxiv_chunks"
BATCH_SIZE = 64
MODEL_NAME = "BAAI/bge-small-en-v1.5"


def load_db_chunks(batch_size=BATCH_SIZE):
    return pd.read_csv(
        CHUNKS_PATH,
        chunksize=batch_size
    )


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def get_collection():
    client = chromadb.PersistentClient(path=DB_PATH)
    return client.get_or_create_collection(name=COLLECTION_NAME)


def embed_chunks(model, collection):
    total_added = 0

    for batch in load_db_chunks():
        batch = batch.dropna(subset=["text"])
        batch = batch[batch["text"].astype(str).str.strip() != ""]

        if batch.empty:
            continue

        ids = batch["chunk_id"].astype(str).tolist()
        documents = batch["text"].astype(str).tolist()

        embeddings = model.encode(documents).tolist()

        metadatas = batch[
            ["paper_id", "chunk_index", "start_idx", "end_idx"]
        ].to_dict(orient="records")

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas
        )

        total_added += len(documents)
        print(f"Added batch. Total added: {total_added}")

    print(f"Finished. Added {total_added} chunks to the collection")
    print(f"Collection count: {collection.count()}")


def main():
    model = load_embedding_model()
    collection = get_collection()
    embed_chunks(model, collection)


if __name__ == "__main__":
    main()