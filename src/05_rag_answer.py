from retrieve_utils import load_embedding_model, load_collection, retrieve_chunks


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
