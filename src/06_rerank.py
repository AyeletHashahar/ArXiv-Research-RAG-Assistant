from sentence_transformers import CrossEncoder
from retrieve_utils import load_embedding_model, load_collection, retrieve_chunks

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_rerank_model():
    return CrossEncoder(RERANK_MODEL)


def cross_encoder(model, query, doc):
    return model.predict([(query, doc)])[0]


def rerank_chunks(model, query, results):
    docs = results["documents"][0]
    ids = results["ids"][0]

    scored = [
        (doc, doc_id, cross_encoder(model, query, doc))
        for doc, doc_id in zip(docs, ids)
    ]
    scored.sort(key=lambda item: item[2], reverse=True)
    return scored


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
