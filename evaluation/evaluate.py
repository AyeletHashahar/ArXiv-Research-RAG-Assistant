"""Retrieval quality metrics + a qualitative end-to-end eval runner.

Quantitative metrics (Recall@K, Precision@K, MRR) score the retriever
against `eval_dataset.EVAL_CASES`. Qualitative evaluation runs full
questions through the whole pipeline (retrieval -> rerank -> LLM) and
prints the answer + sources for manual read-through, which is what you
actually need to judge answer quality, not just chunk overlap.
"""

import logging

from config import RERANK_TOP_N, RETRIEVE_TOP_K
from evaluation.eval_dataset import EVAL_CASES

logger = logging.getLogger(__name__)


def recall_at_k(retrieved_ids, relevant_ids, k):
    if not relevant_ids:
        return None
    retrieved_at_k = set(retrieved_ids[:k])
    return len(retrieved_at_k & set(relevant_ids)) / len(relevant_ids)


def precision_at_k(retrieved_ids, relevant_ids, k):
    retrieved_at_k = retrieved_ids[:k]
    if not retrieved_at_k:
        return 0.0
    hits = sum(1 for chunk_id in retrieved_at_k if chunk_id in relevant_ids)
    return hits / len(retrieved_at_k)


def mrr(retrieved_ids, relevant_ids):
    for rank, chunk_id in enumerate(retrieved_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


def evaluate_retrieval(retriever, eval_cases=EVAL_CASES, k=RETRIEVE_TOP_K):
    """Run every eval case through `retriever` and average Recall@K,
    Precision@K and MRR across cases. Returns the aggregate metrics dict
    plus per-case detail for debugging a specific query.
    """
    if not eval_cases:
        raise ValueError("eval_cases is empty, nothing to evaluate")

    per_case = []
    for case in eval_cases:
        chunks = retriever.retrieve(case["query"], k=k)
        retrieved_ids = [chunk["chunk_id"] for chunk in chunks]

        per_case.append(
            {
                "query": case["query"],
                "recall": recall_at_k(retrieved_ids, case["relevant_chunk_ids"], k),
                "precision": precision_at_k(retrieved_ids, case["relevant_chunk_ids"], k),
                "mrr": mrr(retrieved_ids, case["relevant_chunk_ids"]),
            }
        )

    valid_recalls = [c["recall"] for c in per_case if c["recall"] is not None]
    aggregate = {
        "recall_at_k": sum(valid_recalls) / len(valid_recalls) if valid_recalls else None,
        "precision_at_k": sum(c["precision"] for c in per_case) / len(per_case),
        "mrr": sum(c["mrr"] for c in per_case) / len(per_case),
        "k": k,
        "num_cases": len(per_case),
    }
    return aggregate, per_case


def run_qualitative_eval(pipeline, eval_cases=EVAL_CASES, top_n=RERANK_TOP_N):
    """Run each eval query through the full pipeline and print the answer
    and cited sources for manual inspection."""
    for case in eval_cases:
        query = case["query"]
        try:
            result = pipeline.answer(query, top_n=top_n)
        except Exception as exc:  # LLMError or VectorStoreError, surfaced per-query
            logger.error("Query failed: %r (%s)", query, exc)
            continue

        print(f"\nQ: {query}")
        print(f"A: {result['answer']}")
        if result["sources"]:
            cited = ", ".join(f"[{s['index']}] {s['chunk_id']}" for s in result["sources"])
            print(f"Sources: {cited}")
