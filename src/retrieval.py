"""Query the vector store and return plain, stable dicts.

This is the boundary between ChromaDB's response format (nested lists,
keyed by result-list-index because it supports batched queries we never
use) and the rest of the pipeline. Nothing downstream of Retriever.retrieve
should ever touch a Chroma response directly — that keeps a future swap to
a different vector DB a one-file change instead of a project-wide one.
"""

import logging

from config import RETRIEVE_TOP_K

logger = logging.getLogger(__name__)


class Retriever:
    def __init__(self, embedding_model, collection):
        self._embedding_model = embedding_model
        self._collection = collection

    def retrieve(self, query, k=RETRIEVE_TOP_K):
        if not query or not query.strip():
            raise ValueError("query must be a non-empty string")

        query_embedding = self._embedding_model.encode(query)
        raw = self._collection.query(query_embeddings=[query_embedding], n_results=k)
        chunks = self._normalize(raw)

        if not chunks:
            logger.warning("No results retrieved for query: %r", query)
        return chunks

    @staticmethod
    def _normalize(raw):
        # Chroma nests every field one level for batched queries; we only
        # ever send one query, so we always want index [0].
        documents = raw.get("documents") or [[]]
        ids = raw.get("ids") or [[]]
        metadatas = raw.get("metadatas") or [[]]
        distances = raw.get("distances") or [[]]

        documents, ids, metadatas, distances = documents[0], ids[0], metadatas[0], distances[0]

        return [
            {
                "paper_id": metadata.get("paper_id"),
                "chunk_id": chunk_id,
                "chunk_index": metadata.get("chunk_index"),
                "distance": distance,
                "text": text,
            }
            for text, chunk_id, metadata, distance in zip(documents, ids, metadatas, distances)
        ]
