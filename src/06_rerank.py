from retrieve_utils import (
    load_embedding_model,
    load_collection,
    load_rerank_model,
    retrieve_chunks,
    rerank_chunks,
)


def main(query):
    embedding_model = load_embedding_model()
    collection = load_collection()
    rerank_model = load_rerank_model()

    results = retrieve_chunks(embedding_model, collection, query, k=10)
    reranked = rerank_chunks(rerank_model, query, results)

    for doc, doc_id, score in reranked:
        print(f"score: {score:.4f} | id: {doc_id} | doc: {doc[:100]}")


if __name__ == "__main__":
    query = "What is Lempel-Ziv factorization?"
    main(query)
