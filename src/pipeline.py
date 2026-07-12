"""Wire retrieval -> reranking -> context -> prompt -> LLM into one call.

Each stage is a small, independently testable module (retrieval.py,
reranker.py, context_builder.py, prompt_builder.py, llm.py). RAGPipeline's
only job is to hold them together in the right order — it contains no
retrieval, ranking, or prompting logic of its own.
"""

import logging

from config import RERANK_TOP_N, RETRIEVE_TOP_K
from src.context_builder import build_context
from src.llm import OllamaClient
from src.prompt_builder import build_prompt
from src.reranker import Reranker
from src.retrieval import Retriever
from src.vector_store import VectorStore

logger = logging.getLogger(__name__)


class RAGPipeline:
    def __init__(self, embedding_model, vector_store=None, reranker=None, llm=None):
        self._embedding_model = embedding_model
        vector_store = vector_store or VectorStore()
        collection = vector_store.get_collection()

        self._retriever = Retriever(embedding_model, collection)
        self._reranker = reranker or Reranker()
        self._llm = llm or OllamaClient()

    def answer(self, query, k=RETRIEVE_TOP_K, top_n=RERANK_TOP_N):
        """Run the full pipeline and return {"answer", "sources"}.

        `sources` is the list of {index, paper_id, chunk_id} that actually
        made it into the prompt, so a caller (CLI, UI, tests) can show
        provenance without re-deriving it from the raw chunks.
        """
        chunks = self._retriever.retrieve(query, k=k)
        if not chunks:
            return {
                "answer": "I don't have any relevant information in the knowledge base to answer that.",
                "sources": [],
            }

        reranked = self._reranker.rerank(query, chunks, top_n=top_n)
        context, sources = build_context(reranked)
        messages = build_prompt(query, context)
        answer = self._llm.chat(messages)

        return {"answer": answer, "sources": sources}
