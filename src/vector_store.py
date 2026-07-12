"""Thin wrapper around ChromaDB: collection lifecycle, batched embedding
writes, and duplicate-id safety. This is the only module allowed to import
`chromadb` directly — everything downstream (retrieval.py) talks to plain
Python dicts, never Chroma's response format.
"""

import logging

import chromadb
import pandas as pd

from config import CHROMA_DB_PATH, COLLECTION_NAME, EMBEDDING_BATCH_SIZE

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Raised for any unrecoverable vector store problem (missing
    collection, empty index, etc.) so callers can catch one exception type
    instead of guessing which ChromaDB internals might leak through."""


class VectorStore:
    def __init__(self, db_path=CHROMA_DB_PATH, collection_name=COLLECTION_NAME):
        self._client = chromadb.PersistentClient(path=str(db_path))
        self._collection_name = collection_name
        self._db_path = db_path

    def get_or_create_collection(self):
        return self._client.get_or_create_collection(name=self._collection_name)

    def get_collection(self):
        """Fetch the collection for querying. Raises VectorStoreError with
        an actionable message if the DB hasn't been built yet."""
        try:
            collection = self._client.get_collection(name=self._collection_name)
        except Exception as exc:
            raise VectorStoreError(
                f"Collection '{self._collection_name}' not found at {self._db_path}. "
                "Run `python scripts/build_vector_db.py` first."
            ) from exc

        if collection.count() == 0:
            raise VectorStoreError(
                f"Collection '{self._collection_name}' exists but is empty. "
                "Run `python scripts/build_vector_db.py` to populate it."
            )
        return collection

    def add_chunks(self, collection, ids, documents, embeddings, metadatas):
        """Add a batch of chunks, skipping any ids already present in the
        collection so re-running the build script is idempotent instead of
        erroring out or creating duplicate vectors."""
        existing = set(collection.get(ids=ids)["ids"])
        if existing:
            keep = [idx for idx, doc_id in enumerate(ids) if doc_id not in existing]
            if not keep:
                logger.warning("All %d ids in this batch already exist, skipping", len(ids))
                return 0
            logger.warning("Skipping %d already-indexed ids", len(ids) - len(keep))
            ids = [ids[i] for i in keep]
            documents = [documents[i] for i in keep]
            embeddings = [embeddings[i] for i in keep]
            metadatas = [metadatas[i] for i in keep]

        collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)
        return len(ids)


def _load_chunks_in_batches(chunks_path, batch_size):
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"{chunks_path} not found. Run `python scripts/build_chunks.py` first."
        )
    return pd.read_csv(chunks_path, chunksize=batch_size)


def build_vector_db(embedding_model, chunks_path, batch_size=EMBEDDING_BATCH_SIZE):
    """Embed every chunk in `chunks_path` and write it into the vector
    store, in batches so memory stays bounded for large corpora."""
    store = VectorStore()
    collection = store.get_or_create_collection()

    total_added = 0
    for batch in _load_chunks_in_batches(chunks_path, batch_size):
        batch = batch.dropna(subset=["text"])
        batch = batch[batch["text"].astype(str).str.strip() != ""]
        if batch.empty:
            continue

        ids = batch["chunk_id"].astype(str).tolist()
        documents = batch["text"].astype(str).tolist()
        embeddings = embedding_model.encode(documents).tolist()
        metadatas = batch[["paper_id", "chunk_index", "start_idx", "end_idx"]].to_dict(
            orient="records"
        )

        added = store.add_chunks(collection, ids, documents, embeddings, metadatas)
        total_added += added
        logger.info("Added batch of %d chunks (total added: %d)", added, total_added)

    logger.info("Finished. Added %d chunks. Collection count: %d", total_added, collection.count())
    return total_added
