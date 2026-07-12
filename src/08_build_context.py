import requests

from retrieve_utils import (
    load_embedding_model,
    load_collection,
    load_rerank_model,
    retrieve_chunks,
    rerank_chunks,
)

OLLAMA_URL = "http://localhost:11434/api/chat"
LLM_MODEL = "qwen2.5"

RETRIEVE_K = 10
CONTEXT_TOP_N = 5

SYSTEM_PROMPT = (
    "You are a medical research assistant. Answer the user's question using "
    "ONLY the information in the provided context. If the context does not "
    "contain enough information to answer, say so instead of guessing. "
    "Do not provide medical diagnoses or treatment advice; summarize what "
    "the source material says."
)


def build_context(reranked_chunks, top_n=CONTEXT_TOP_N):
    top_chunks = reranked_chunks[:top_n]
    return "\n\n".join(
        f"[{i + 1}] {doc}" for i, (doc, _doc_id, _score) in enumerate(top_chunks)
    )


def build_prompt(query, context):
    return (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above, and cite sources like [1], [2] "
        "where relevant."
    )


def generate_answer(query, context):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_prompt(query, context)},
            ],
            "stream": False,
        },
    )
    response.raise_for_status()
    return response.json()["message"]["content"]


def answer_question(embedding_model, collection, rerank_model, query):
    results = retrieve_chunks(embedding_model, collection, query, k=RETRIEVE_K)
    reranked = rerank_chunks(rerank_model, query, results)
    context = build_context(reranked)
    return generate_answer(query, context)


def main():
    print("Loading models...")
    embedding_model = load_embedding_model()
    collection = load_collection()
    rerank_model = load_rerank_model()
    print("Ready. Type a question (or 'exit' to quit).")

    while True:
        query = input("\nQuestion: ").strip()
        if query.lower() in ("exit", "quit"):
            break
        if not query:
            continue

        answer = answer_question(embedding_model, collection, rerank_model, query)
        print(f"\nAnswer:\n{answer}")


if __name__ == "__main__":
    main()
