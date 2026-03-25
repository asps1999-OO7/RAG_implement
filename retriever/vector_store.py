"""
Vector Store: ChromaDB interface for both writes and reads.
"""

import os
import logging
from typing import List
import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv
from retriever.embedder import Chunk

load_dotenv()
log = logging.getLogger(__name__)

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_store")
COLLECTION_NAME = "documents"


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=CHROMA_DIR,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        log.info(f"ChromaDB ready. Docs in store: {self.collection.count()}")

    def upsert(self, chunks: List[Chunk], embeddings: List[List[float]]):
        self.collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[{
                "source": c.source,
                "page": c.page,
                "chunk_index": c.chunk_index,
            } for c in chunks],
        )
        log.info(f"Upserted {len(chunks)} chunks. Total: {self.collection.count()}")

    def query(self, query_embedding: List[float], top_k: int = 5, source_filter: str = None) -> List[dict]:
        where = {"source": source_filter} if source_filter else None

        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.collection.count() or 1),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            hits.append({
                "text": doc,
                "source": meta["source"],
                "page": meta["page"],
                "distance": dist,
            })
        return hits

    def list_sources(self) -> List[str]:
        if self.collection.count() == 0:
            return []
        all_meta = self.collection.get(include=["metadatas"])["metadatas"]
        return list({m["source"] for m in all_meta})
