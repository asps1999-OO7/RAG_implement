# ⚡ Kafka-RAG: Real-Time Document Intelligence Pipeline

> A production-grade RAG (Retrieval Augmented Generation) system where the knowledge base is a **live streaming pipeline** — drop a PDF, ask questions within seconds.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://python.org)
[![Kafka](https://img.shields.io/badge/Apache_Kafka-2.3.0-black.svg)](https://kafka.apache.org)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4.24-purple.svg)](https://www.trychroma.com)
[![Groq](https://img.shields.io/badge/Groq-Llama3-orange.svg)](https://groq.com)
[![NVIDIA](https://img.shields.io/badge/NVIDIA-Nemotron_Reranker-green.svg)](https://build.nvidia.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32.0-red.svg)](https://streamlit.io)

---

## 📌 What Problem Does This Solve?

Enterprise teams accumulate thousands of PDFs — reports, policies, manuals, research papers. Finding specific information inside them is slow and manual. This system lets you:

1. **Drop any PDF** into a folder
2. **Ask natural language questions** about it within seconds
3. **Get cited, grounded answers** — with source document and page number

The key differentiator: the knowledge base is not a static batch load. It is a **live Kafka-driven data pipeline**. Documents flow in continuously, get indexed automatically, and become queryable immediately — without restarting anything.

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        INGESTION LAYER                          │
│                                                                 │
│   /docs folder                                                  │
│       │                                                         │
│       │  File dropped                                           │
│       ▼                                                         │
│   ┌─────────────────────┐                                       │
│   │  document_producer  │  ← Watchdog monitors folder          │
│   │  .py                │    Publishes JSON event to Kafka     │
│   └──────────┬──────────┘    Partition key = filename          │
│              │                                                  │
└──────────────┼──────────────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────────────┐
│                        MESSAGE BROKER                           │
│                                                                 │
│              Apache Kafka                                       │
│                                                                 │
│   Topic: document-events                                        │
│   Partitions: 3  (3 consumers can run in parallel)             │
│   Partition key: filename  (ordering per document guaranteed)  │
│   Retention: 24-48 hours   (replay on consumer failure)        │
│                                                                 │
└──────────────────────┬──────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                      PROCESSING LAYER                           │
│                                                                 │
│   ┌──────────────────────────┐                                  │
│   │  document_consumer.py   │  ← Runs continuously            │
│   │                          │    Manual commit only after     │
│   │  1. PyMuPDF              │    successful ChromaDB write    │
│   │     └─ Parse PDF text    │                                  │
│   │        + page numbers    │                                  │
│   │                          │                                  │
│   │  2. Chunker (custom)     │                                  │
│   │     └─ 512 char chunks   │                                  │
│   │        50 char overlap   │                                  │
│   │        Recursive split   │                                  │
│   │                          │                                  │
│   │  3. PDFEmbedder          │                                  │
│   │     └─ sentence-trans    │                                  │
│   │        all-MiniLM-L6-v2  │                                  │
│   │        → 384-dim vector  │                                  │
│   └──────────┬───────────────┘                                  │
│              │                                                  │
│              ▼                                                  │
│   ┌──────────────────────────┐                                  │
│   │  ChromaDB                │  ← Persistent local vector DB  │
│   │  (chroma_store/)         │    HNSW index, cosine metric   │
│   │                          │    Upsert = no duplicates      │
│   │  Stores per chunk:       │                                  │
│   │  • chunk_id (unique)     │                                  │
│   │  • embedding (384 nums)  │                                  │
│   │  • original text         │                                  │
│   │  • source + page         │                                  │
│   └──────────────────────────┘                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────────┐
│                       QUERY LAYER                               │
│                                                                 │
│   User types question in Streamlit UI                           │
│       │                                                         │
│       ▼                                                         │
│   [1] Embed question → 384-dim vector (same model as chunks)   │
│       │                                                         │
│       ▼                                                         │
│   [2] ChromaDB HNSW search → top 5 similar chunks             │
│       │         (O(log n), approximate nearest neighbour)      │
│       │                                                         │
│       ▼                                                         │
│   [3] NVIDIA Nemotron Reranker API                              │
│       │  Cross-encoder scores each (question, chunk) pair      │
│       │  Keeps top 3 by logit score                            │
│       │  More precise than vector similarity alone             │
│       │                                                         │
│       ▼                                                         │
│   [4] Build grounded prompt                                     │
│       │  "Answer ONLY from context below."                     │
│       │  "Cite source and page."                               │
│       │  + top 3 chunks injected as context                   │
│       │                                                         │
│       ▼                                                         │
│   [5] Groq API → Llama3 (stream=True, temperature=0.1)        │
│       │  Token-by-token response to UI                         │
│       │                                                         │
│       ▼                                                         │
│   Answer + Source Citations displayed in chat UI               │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔍 Two-Stage Retrieval — Why It Matters

Most RAG tutorials use only vector similarity. This project uses **two stages**:

```
Stage 1: Vector Search (ChromaDB)          Stage 2: Reranking (NVIDIA Nemotron)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Type:    Bi-encoder                         Type:    Cross-encoder
Speed:   Fast (precomputed embeddings)      Speed:   Slower (computed at query time)
Goal:    High RECALL                        Goal:    High PRECISION
Method:  Cosine similarity of vectors       Method:  Joint scoring of (query, chunk)
Returns: Top 5 candidates                  Returns: Top 3 final chunks
                │                                          │
                └──────────────► feeds into ──────────────┘

Analogy: Google Search (wide net)  +  You manually reading top results (careful pick)
```

---

## 📁 Project Structure

```
kafka-rag/
│
├── producer/
│   └── document_producer.py     # Watchdog → Kafka publisher
│
├── consumer/
│   └── document_consumer.py     # Kafka consumer → ChromaDB indexer
│
├── retriever/
│   ├── __init__.py
│   ├── embedder.py              # PDF parser + chunker + embedding model
│   ├── vector_store.py          # ChromaDB read/write interface
│   └── reranker.py              # NVIDIA Nemotron API reranker
│
├── ui/
│   └── app.py                   # Streamlit chat interface
│
├── scripts/
│   └── evaluate.py              # RAGAs evaluation (faithfulness, relevancy)
│
├── docs/                        # ← Drop your PDFs here
│
├── chroma_store/                # Auto-created: persistent vector DB
│
├── docker-compose.yml           # Kafka + Zookeeper
├── requirements.txt
├── .env.example                 # Copy to .env and fill your keys
└── .gitignore
```

---

## ⚙️ Component Deep Dive

### 1. Producer — `document_producer.py`

Watches the `/docs` folder using **Watchdog** (a filesystem monitoring library). The moment a PDF appears, it publishes a JSON event to Kafka.

```python
# What gets published to Kafka
payload = {
    "filename": "report.pdf",
    "filepath": "/docs/report.pdf",
    "event_type": "new_document",
    "timestamp": 1741234567.0
}
```

**Key design decisions:**
- `time.sleep(0.5)` — waits for the file to finish copying before publishing
- `key=path.name` — partition key ensures same file always routes to same partition (ordering guarantee)
- `self.seen` set — deduplicates rapid filesystem events (OS sometimes fires twice for one file)

---

### 2. Kafka — Message Broker

```
Topic: document-events
│
├── Partition 0  ← filenames hashing to 0
├── Partition 1  ← filenames hashing to 1
└── Partition 2  ← filenames hashing to 2

Consumer Group: rag-indexer-group
└── Offset stored in __consumer_offsets (internal Kafka topic)
    → Consumer restarts from exact position, nothing replayed or lost
```

**Why Kafka and not a direct function call?**

| Without Kafka | With Kafka |
|---|---|
| Consumer crash = event lost | Consumer crash = message replays |
| Producer blocked if consumer slow | Producer and consumer fully decoupled |
| No parallelism | 3 partitions = 3 parallel consumers |
| No audit trail | Full message history for 24-48h |

---

### 3. Consumer — `document_consumer.py`

The most critical component. Runs forever, processing events one by one.

```python
Consumer({
    "auto.offset.reset": "earliest",   # First run: start from beginning
    "enable.auto.commit": False,        # CRITICAL: manual commit only
})

# The commit only happens AFTER successful ChromaDB write
process_message(payload, embedder, store)
consumer.commit(msg)   ← if this line is not reached, message replays
```

**The manual commit pattern** is what makes this production-grade. Auto-commit would silently lose messages on crash. Manual commit guarantees exactly-once processing.

---

### 4. Embedder — `retriever/embedder.py`

Three steps: Parse → Chunk → Embed

**Parsing:**
```
PyMuPDF (fitz) opens PDF
→ Extracts raw text per page
→ Keeps page number for citations
→ Skips empty pages
```

**Chunking (written from scratch — no LangChain):**
```
CHUNK_SIZE = 512 characters
CHUNK_OVERLAP = 50 characters
Separator hierarchy: ["\n\n", "\n", ". ", " ", ""]

Tries paragraph breaks first → then sentences → then words → then characters
Overlap ensures sentences at chunk boundaries appear fully in at least one chunk
```

**Why 512 characters?**
- Large enough: ~100-150 words, enough context for meaningful retrieval
- Small enough: topically focused, no noise from unrelated content

**Embedding:**
```python
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
# 384-dimensional vectors
# Runs locally on CPU — no GPU needed
# ~80MB model size
embeddings = model.encode(texts, batch_size=32).tolist()
# .tolist() converts numpy array → Python list (ChromaDB requires lists)
```

---

### 5. Vector Store — `retriever/vector_store.py`

```python
# Storage: persistent on disk at ./chroma_store
# Similarity: cosine (angle between vectors, not euclidean distance)
# Index: HNSW — O(log n) approximate nearest neighbour

collection.upsert(
    ids=[chunk_id],          # "filename_p3_c2" — unique per chunk
    embeddings=[...],        # 384 numbers
    documents=[text],        # original chunk text
    metadatas=[{source, page}]  # for citations in UI
)
```

**Why upsert not insert?**
Re-dropping the same PDF overwrites existing chunks instead of creating duplicates. Safe to re-run anytime.

**HNSW (Hierarchical Navigable Small World):**
A graph where each vector connects to nearest neighbours at multiple layers. Search starts at sparse top layer and navigates down to dense bottom layer. 99% accurate at a fraction of brute-force cost.

---

### 6. Reranker — `retriever/reranker.py`

```
Request to NVIDIA API:
POST https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking

{
  "query": {"text": "user's question"},
  "passages": [{"text": chunk1}, {"text": chunk2}, ...],  ← top 5 from ChromaDB
  "truncate": "END"
}

Response:
{
  "rankings": [
    {"index": 2, "logit": 4.21},   ← chunk 2 is most relevant
    {"index": 0, "logit": 2.87},
    {"index": 4, "logit": 1.33},
    ...
  ]
}
```

**Logit score** = raw neural network output. Higher = more relevant. Not a probability. Typical range -5 to +5.

**Fallback:** If NVIDIA API is unavailable, returns top N by original vector similarity order. System stays functional.

---

### 7. UI — `ui/app.py`

```python
@st.cache_resource          # Loads embedding model ONCE
def load_pipeline():        # Reused across all user messages
    return PDFEmbedder(), VectorStore(), Reranker()

# Streaming response — tokens appear live as LLM generates them
response_text = st.write_stream(
    chunk.choices[0].delta.content or ""
    for chunk in result
    if chunk.choices[0].delta.content
)
```

**The grounded prompt:**
```
You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information."
Always cite which source and page your answer comes from.

CONTEXT:
[Source 1: report.pdf, Page 3]
...chunk text...

[Source 2: report.pdf, Page 7]
...chunk text...

QUESTION: {user_question}
ANSWER:
```

`temperature=0.1` keeps answers factual and consistent. `stream=True` gives live typing effect.

---

## 🚀 Setup & Running

### Prerequisites
- Python 3.12
- Docker (for Kafka)
- Groq API key (free at [console.groq.com](https://console.groq.com))
- NVIDIA API key (free at [build.nvidia.com](https://build.nvidia.com))

### Step 1 — Start Kafka
```bash
docker compose up -d
```

### Step 2 — Install dependencies
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install CPU-only PyTorch first (avoids downloading 3GB CUDA build)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining dependencies
pip install -r requirements.txt
```

### Step 3 — Configure environment
```bash
cp .env.example .env
# Edit .env and fill in your API keys
```

```env
GROQ_API_KEY=gsk_xxxxxxxxxxxxxxxxxxxx
NVIDIA_API_KEY=nvapi-xxxxxxxxxxxxxxxxxxxx
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
KAFKA_TOPIC=document-events
CHROMA_PERSIST_DIR=./chroma_store
DOCS_WATCH_DIR=./docs
```

### Step 4 — Run (3 terminals)

**Terminal 1 — Pipeline worker (must start first)**
```bash
source venv/bin/activate
python consumer/document_consumer.py
```
Expected output:
```
2026-03-28 [CONSUMER] Loading embedding model: sentence-transformers/all-MiniLM-L6-v2
2026-03-28 [CONSUMER] Embedding model ready.
2026-03-28 [CONSUMER] ChromaDB ready. Docs in store: 0
2026-03-28 [CONSUMER] Subscribed to document-events. Waiting for documents...
```

**Terminal 2 — Folder watcher**
```bash
source venv/bin/activate
python producer/document_producer.py
```
Expected output:
```
2026-03-28 [PRODUCER] Watching ./docs for new PDFs... (Ctrl+C to stop)
```

**Terminal 3 — Chat UI**
```bash
source venv/bin/activate
streamlit run ui/app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### Step 5 — Test it
```bash
# Drop any PDF into the docs folder
cp any_document.pdf docs/

# Watch Terminal 1 — you'll see:
# [CONSUMER] Processing: any_document.pdf
# [CONSUMER] Parsed 12 pages from any_document.pdf
# [CONSUMER] Generated 47 chunks
# [CONSUMER] Embedded 47 chunks.
# [CONSUMER] Upserted 47 chunks. Total: 47

# Now open localhost:8501 and ask a question about the document
```

---

## 📊 Evaluation

Run RAGAs evaluation after indexing documents:

```bash
python scripts/evaluate.py
```

**Metrics measured:**

| Metric | What it checks |
|---|---|
| **Faithfulness** | Is the answer grounded in retrieved context? (catches hallucination) |
| **Answer Relevancy** | Does the answer address the question asked? |
| **Context Precision** | Were the retrieved chunks actually relevant? |

Example output:
```
── RAGAs Evaluation Results ──
Faithfulness:      0.91
Answer Relevancy:  0.87
Context Precision: 0.84
──────────────────────────────
```

---

## 🛑 Stopping the Project

```bash
# Stop all processes
pkill -f "document_consumer"
pkill -f "document_producer"
pkill -f "streamlit"

# Stop Kafka
docker compose down

# Stop Docker daemon completely
sudo systemctl stop docker
sudo systemctl stop docker.socket
```

---

## 🔬 What Makes This Different From Tutorial RAG

| Feature | Tutorial RAG | This Project |
|---|---|---|
| Ingestion | Static batch load | Real-time Kafka streaming |
| Chunking | LangChain wrapper | Written from scratch |
| Retrieval | Vector similarity only | Two-stage: vector + cross-encoder reranking |
| Reranker | Local cross-encoder | NVIDIA Nemotron API (GPU-accelerated) |
| LLM | OpenAI API | Groq (free, fast) |
| Evaluation | None | RAGAs metrics |
| Fault tolerance | None | Kafka manual commit + ChromaDB upsert |

---

## 📈 Production Scaling Path

```
Current (Portfolio)              Production Scale
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Streamlit UI            →        FastAPI + async endpoints
Single consumer         →        3 consumers (match partitions)
ChromaDB local          →        Qdrant / Weaviate (distributed)
Dense retrieval only    →        Hybrid: dense + BM25 sparse
No caching              →        Redis for frequent queries
CPU embedding           →        GPU instance / NVIDIA embed API
No auth                 →        JWT + API gateway
```

---

## 🤝 Tech Stack

| Component | Technology | Why |
|---|---|---|
| Message broker | Apache Kafka | Fault tolerance, decoupling, replay |
| PDF parsing | PyMuPDF (fitz) | Fast, accurate, page metadata |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | Free, local, 384-dim |
| Vector DB | ChromaDB | Local persistent, HNSW, cosine |
| Reranker | NVIDIA Llama-Nemotron-Rerank-1b-v2 | Purpose-built for retrieval scoring |
| LLM | Groq → Llama3-70b | Free tier, fast inference, streaming |
| UI | Streamlit | Rapid prototyping, cache_resource |
| Evaluation | RAGAs | Faithfulness, relevancy, precision |
| Orchestration | Watchdog | Lightweight folder monitoring |

---

## 👤 Author

**Puneet Saran** — Data Platform Engineer  
[puneetsaran.netlify.app](https://puneetsaran.netlify.app) · [puneet19.saran@gmail.com](mailto:puneet19.saran@gmail.com)

---

## 📄 License

MIT License — free to use, modify, and distribute.
