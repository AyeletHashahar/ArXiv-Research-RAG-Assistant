"""Hand-labeled (query, relevant_chunk_ids) pairs used to score retrieval.

LIMITATION: the source corpus (arxiv-summarization) has no built-in QA
labels, so these labels were bootstrapped by keyword search — for each
query we located a chunk whose text contains the corresponding term via
`grep`/`pandas.str.contains` and treated it as the single relevant chunk.
This is a weak-labeling technique, fine for smoke-testing retrieval
quality during development, but it understates real recall (a chunk can be
relevant without containing the exact keyword). A production system should
replace this with human-graded relevance judgments or an LLM-as-judge
labeling pass over a larger sample.
"""

EVAL_CASES = [
    {"query": "What is Lempel-Ziv factorization?", "relevant_chunk_ids": ["739_0"]},
    {"query": "What is stringology?", "relevant_chunk_ids": ["739_0"]},
    {
        "query": "How does graph coloring apply to state estimation attacks?",
        "relevant_chunk_ids": ["164_0"],
    },
    {
        "query": "How is reinforcement learning used with Q-learning?",
        "relevant_chunk_ids": ["362_3"],
    },
    {"query": "What is a fuzzy min-max neural network?", "relevant_chunk_ids": ["7_3"]},
]
