from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb

MODEL_NAME = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def load_collection():
    client = chromadb.PersistentClient(path="chroma_db")
    return client.get_collection(name="arxiv_chunks")


def retrieve_chunks(model, collection, query, k=5):
    query_embedding = model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k
    )
    return results


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
