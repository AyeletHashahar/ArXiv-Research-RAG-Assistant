"""Central configuration for the RAG pipeline.

Every tunable parameter lives here so behavior can be changed in one place
instead of hunting through source files for magic numbers.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# --- Paths -------------------------------------------------------------
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "arxiv-summarization.csv"
CHUNKS_PATH = DATA_DIR / "chunks.csv"
CHROMA_DB_PATH = BASE_DIR / "chroma_db"

# --- Source dataset ------------------------------------------------------
DATASET_NAME = "ccdv/arxiv-summarization"
DATASET_SPLIT = "train"
DATASET_SAMPLE_SIZE = 1000

# --- Chunking ------------------------------------------------------------
CHUNK_SIZE = 5000
CHUNK_OVERLAP = 500
CHUNK_WRITE_BATCH_SIZE = 1000

# --- Vector store ----------------------------------------------------------
COLLECTION_NAME = "arxiv_chunks"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_BATCH_SIZE = 64

# --- Retrieval -------------------------------------------------------------
RETRIEVE_TOP_K = 10

# --- Reranking ---------------------------------------------------------
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_N = 5

# --- Context building ----------------------------------------------------
# Max characters of chunk text included in a single prompt. Keeps the
# prompt within the LLM's context window regardless of how many/large the
# reranked chunks are.
CONTEXT_MAX_CHARS = 6000

# --- LLM (Ollama) --------------------------------------------------------
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "qwen2.5"
LLM_REQUEST_TIMEOUT_SECONDS = 60

# --- Logging -------------------------------------------------------------
LOG_LEVEL = "INFO"
