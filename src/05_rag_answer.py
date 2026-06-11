from sentence_transformers import SentenceTransformer
import chromadb

MODEL_NAME = "BAAI/bge-small-en-v1.5"

CHUNKS_PATH = "data/chunks.csv"
BATCH_SIZE = 64


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

def answer_question(model, collection, query, k=5):
    results = retrieve_chunks(model, collection, query, k)
    return results



def main():
    model = load_embedding_model()
    collection = load_collection()
    query = "What is Lempel-Ziv factorization?"
    results = answer_question(model, collection, query, k=5)
    print(f"query: {query}")
    print(f"results: {results}")
    query = "What is LZ77?"
    results = answer_question(model, collection, query, k=5)
    print(f"query: {query}")
    print(f"results: {results}")
    query = "How is data compression performed?"
    results = answer_question(model, collection, query, k=5)
    print(f"query: {query}")
    print(f"results: {results}")
    query = "What are runs in string algorithms?"
    results = answer_question(model, collection, query, k=5)
    print(f"query: {query}")
    print(f"results: {results}")

    query = "What is stringology?"
    results = answer_question(model, collection, query, k=5)
    print(f"query: {query}")
    print(f"results: {results}")



if __name__ == "__main__":
    main()
