"""Cross-encoder reranking over retrieved chunks."""

import logging

from sentence_transformers import CrossEncoder

from config import RERANK_MODEL, RERANK_TOP_N

logger = logging.getLogger(__name__)


class Reranker:
    def __init__(self, model_name=RERANK_MODEL):
        self._model = CrossEncoder(model_name)

    def rerank(self, query, chunks, top_n=RERANK_TOP_N):
        """Score every chunk against `query` in a single batched
        CrossEncoder.predict() call, then return the top_n chunks sorted
        by descending relevance with a `rerank_score` field attached.
        """
        if not chunks:
            return []

        pairs = [(query, chunk["text"]) for chunk in chunks]
        scores = self._model.predict(pairs)

        scored = [
            {**chunk, "rerank_score": float(score)} for chunk, score in zip(chunks, scores)
        ]
        scored.sort(key=lambda chunk: chunk["rerank_score"], reverse=True)
        return scored[:top_n]
