"""Assemble a prompt-ready context block from reranked chunks."""

import logging

from config import CONTEXT_MAX_CHARS

logger = logging.getLogger(__name__)


def build_context(chunks, max_chars=CONTEXT_MAX_CHARS):
    """Join reranked chunks (already in relevance order) into one text
    block, each tagged with a numbered source header, stopping before
    `max_chars` so the prompt can't blow past the LLM's context window.

    Returns (context_text, sources) where `sources` maps each [N] marker
    back to its paper_id/chunk_id, so the LLM's citations can be verified
    or rendered as links by a caller (e.g. a UI).
    """
    if not chunks:
        return "", []

    parts = []
    sources = []
    total_chars = 0

    for i, chunk in enumerate(chunks, start=1):
        header = f"[Source {i} | paper_id={chunk['paper_id']} | chunk_id={chunk['chunk_id']}]"
        block = f"{header}\n{chunk['text']}"

        # Always include at least one source, even if it alone exceeds the
        # budget, so a single long chunk can't produce an empty context.
        if parts and total_chars + len(block) > max_chars:
            logger.debug("Context budget reached after %d/%d chunks", i - 1, len(chunks))
            break

        parts.append(block)
        total_chars += len(block)
        sources.append({"index": i, "paper_id": chunk["paper_id"], "chunk_id": chunk["chunk_id"]})

    return "\n\n".join(parts), sources
