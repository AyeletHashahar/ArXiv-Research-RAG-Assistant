"""Build the chat messages sent to the LLM."""

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions strictly from the "
    "provided context.\n"
    "Rules:\n"
    "1. Answer ONLY using information found in the context below.\n"
    "2. Never invent facts, sources, or details that are not in the context.\n"
    "3. If the context does not contain enough information to answer, say so "
    "explicitly instead of guessing.\n"
    "4. Cite the source(s) you used for each claim using their marker, e.g. "
    "[Source 1], [Source 2]."
)


def build_prompt(query, context):
    """Return an Ollama/OpenAI-style chat messages list: [system, user]."""
    if not context:
        user_content = (
            f"Question: {query}\n\n"
            "No relevant context was found in the knowledge base. State that "
            "you cannot answer the question from the available sources."
        )
    else:
        user_content = f"Context:\n{context}\n\nQuestion: {query}"

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
