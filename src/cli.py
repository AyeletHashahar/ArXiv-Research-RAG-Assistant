"""Interactive question-answering loop over the RAG pipeline."""

import logging

from config import EMBEDDING_MODEL
from src.llm import LLMError
from src.logging_config import configure_logging
from src.pipeline import RAGPipeline
from src.vector_store import VectorStoreError

logger = logging.getLogger(__name__)


def run():
    configure_logging()

    from sentence_transformers import SentenceTransformer

    logger.info("Loading embedding model (%s)...", EMBEDDING_MODEL)
    embedding_model = SentenceTransformer(EMBEDDING_MODEL)

    try:
        pipeline = RAGPipeline(embedding_model)
    except VectorStoreError as exc:
        logger.error(str(exc))
        return

    print("Ready. Type a question (or 'exit' to quit).")

    while True:
        try:
            query = input("\nQuestion: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        try:
            result = pipeline.answer(query)
        except LLMError as exc:
            print(f"\n[LLM error] {exc}")
            continue

        print(f"\nAnswer:\n{result['answer']}")
        if result["sources"]:
            cited = ", ".join(f"[{s['index']}] {s['chunk_id']}" for s in result["sources"])
            print(f"\nSources: {cited}")


if __name__ == "__main__":
    run()
