# 📚 ArXiv Research Assistant - Retrieval-Augmented Generation (RAG)

A production-style Retrieval-Augmented Generation (RAG) system built from scratch using modern NLP and Information Retrieval techniques.

This project demonstrates the complete RAG pipeline without relying on frameworks such as LangChain, providing a deeper understanding of every stage in the retrieval process.

---

## 🚀 Features

- 📄 Load research papers from the ArXiv dataset
- ✂️ Document chunking with configurable overlap
- 🧠 Dense embeddings using **BAAI/bge-small-en-v1.5**
- 🗂️ Persistent vector database using **ChromaDB**
- 🔍 Semantic retrieval (Top-K search)
- 🎯 Cross-Encoder reranking
- 🤖 Local LLM inference using **Ollama (Qwen2.5)**
- 💬 Context-aware question answering
- ⚡ Batch processing for efficient indexing

---

# 🏗 Architecture

```
Research Papers
        │
        ▼
Document Loader
        │
        ▼
Chunking
        │
        ▼
Embedding Model
 (BAAI/bge-small-en-v1.5)
        │
        ▼
ChromaDB
(Vector Database)
        │
        ▼
Semantic Retrieval
        │
        ▼
Cross Encoder
Re-ranking
        │
        ▼
Context Builder
        │
        ▼
Prompt Builder
        │
        ▼
Qwen2.5 (Ollama)
        │
        ▼
Final Answer
```

---

# 📂 Project Structure

```
medical-rag-assistant/

│
├── data/
│   ├── arxiv-summarization.csv
│   └── chunks.csv
│
├── chroma_db/
│
├── src/
│   ├── 01_load_data.py
│   ├── 02_create_chunks.py
│   ├── 03_build_vector_db.py
│   ├── retrieve_utils.py
│   ├── 06_rerank.py
│   ├── 08_build_context.py
│   └── 05_rag_answer.py
│
├── requirements.txt
└── README.md
```

---

# ⚙️ Technologies

- Python
- Pandas
- HuggingFace Datasets
- SentenceTransformers
- ChromaDB
- Ollama
- Qwen2.5
- Cross Encoder
- Vector Search

---

# 📖 Retrieval Pipeline

## 1. Load Dataset

Research papers are downloaded from the HuggingFace ArXiv dataset.

---

## 2. Chunking

Large documents are divided into overlapping chunks.

Current configuration:

- Chunk Size = 5000 characters
- Overlap = 500 characters

---

## 3. Embeddings

Each chunk is converted into a dense embedding using

```
BAAI/bge-small-en-v1.5
```

---

## 4. Vector Database

Embeddings are stored inside a persistent ChromaDB collection.

---

## 5. Retrieval

For every user question:

- Convert the query into an embedding
- Search the vector database
- Retrieve the Top-K most similar chunks

---

## 6. Re-ranking

Retrieved chunks are re-ranked using

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

to improve precision before sending context to the LLM.

---

## 7. Context Construction

The highest ranked chunks are combined into a single context.

---

## 8. Answer Generation

The context is passed to a locally running

```
Qwen2.5
```

model via Ollama to generate the final grounded answer.

---

# ▶️ Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/medical-rag-assistant.git
cd medical-rag-assistant
```

Create a virtual environment

```bash
python -m venv venv
```

Activate it

Windows

```bash
venv\Scripts\activate
```

Linux / Mac

```bash
source venv/bin/activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# 📥 Download the LLM

Install Ollama

https://ollama.com

Download the model

```bash
ollama pull qwen2.5
```

Run it

```bash
ollama run qwen2.5
```

---

# ▶️ Running the Project

### Load Dataset

```bash
python src/01_load_data.py
```

### Create Chunks

```bash
python src/02_create_chunks.py
```

### Build Vector Database

```bash
python src/03_build_vector_db.py
```

### Run Retrieval

```bash
python src/retrieve_utils.py
```

### Run Re-ranking

```bash
python src/06_rerank.py
```

### Ask Questions

```bash
python src/05_rag_answer.py
```

---

# 💡 Example Questions

```
What is Retrieval-Augmented Generation?

Explain Lempel-Ziv factorization.

What are runs in string algorithms?

How is data compression performed?
```

---

# 🎯 Learning Goals

This project was intentionally implemented without high-level RAG frameworks (e.g., LangChain) to better understand every stage of the retrieval pipeline.

Implemented concepts include:

- Document Chunking
- Dense Embeddings
- Vector Databases
- Semantic Retrieval
- Cross-Encoder Re-ranking
- Prompt Engineering
- Context Construction
- Local LLM Inference

---

# 🚧 Future Improvements

- Hybrid Retrieval (BM25 + Dense Retrieval)
- Query Expansion
- Metadata Filtering
- Streaming Responses
- Evaluation (Recall@K, MRR)
- Hallucination Detection
- Conversation Memory
- Multi-Agent RAG
- Streamlit Chat UI

---

# 📄 License

This project was created for educational purposes to study modern Retrieval-Augmented Generation systems and Information Retrieval techniques.
