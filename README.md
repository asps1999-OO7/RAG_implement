       # Kafka-RAG: Real-Time Knowledge Pipeline

A RAG system where the knowledge base is a **live streaming pipeline** — drop a PDF into `/docs` and it becomes queryable within seconds, without restarting anything.

## Architecture

```
/docs folder (new PDF dropped)
        │
        ▼
 Producer (watchdog)
 watches folder for new files
        │
        ▼
 Kafka Topic: document-events
 partition key: filename
        │
        ▼
 Consumer (always running)
 - parses PDF → PyMuPDF
 - chunks text (512 chars, 50 overlap)
 - embeds chunks → sentence-transformers
 - writes vectors → ChromaDB
        │
        ▼
 Streamlit UI
 - user types question
 - query embedded → similarity search (ChromaDB)
 - reranked → NVIDIA Nemotron Reranker API
 - top chunks assembled into prompt
 - Groq LLM (Llama3) generates streamed answer
 - source citations shown
```

## Stack

| Component | Tool |
|---|---|
| Message broker | Apache Kafka (Docker) |
| PDF parsing | PyMuPDF |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Vector DB | ChromaDB (local persistent) |
| Reranker | NVIDIA Llama-Nemotron-Rerank-1b-v2 API |
| LLM | Groq API — Llama3-8b (free tier) |
| UI | Streamlit |
| Evaluation | RAGAs |

## Setup

```bash
# 1. Start Kafka
docker-compose up -d

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env from example and fill in your keys
cp .env.example .env

# 4. Terminal 1 — pipeline worker
python consumer/document_consumer.py

# 5. Terminal 2 — folder watcher
python producer/document_producer.py

# 6. Terminal 3 — chat UI
streamlit run ui/app.py

# 7. Drop any PDF into /docs — ask questions instantly
```

## API Keys Needed

- **Groq** (free): https://console.groq.com
- **NVIDIA NIM** (free): https://build.nvidia.com

## Evaluation

```bash
python scripts/evaluate.py
```

Measures faithfulness, answer relevancy, and context precision using RAGAs.
