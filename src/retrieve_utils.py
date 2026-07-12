from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "BAAI/bge-small-en-v1.5"


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
