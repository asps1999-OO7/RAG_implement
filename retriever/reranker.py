"""
Reranker: NVIDIA Llama-Nemotron Rerank API.

Upgraded from local cross-encoder to NVIDIA's GPU-accelerated
reranking model — purpose-built for passage retrieval scoring.

Why this is better than a local cross-encoder:
- Trained specifically for retrieval reranking
- GPU-accelerated on NVIDIA infrastructure
- No local model download needed
- Falls back gracefully to vector similarity if API fails
"""

import os
import logging
import requests
from typing import List
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
RERANK_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"


class Reranker:
    def __init__(self):
        if not NVIDIA_API_KEY:
            raise ValueError("NVIDIA_API_KEY not set in .env")
        self.headers = {
            "Authorization": f"Bearer {NVIDIA_API_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        log.info("NVIDIA Nemotron Reranker ready.")

    def rerank(self, query: str, hits: List[dict], top_n: int = 3) -> List[dict]:
        """
        Rerank retrieved chunks using NVIDIA Nemotron API.
        Adds 'rerank_score' to each hit, returns top_n by score.
        """
        if not hits:
            return []

        passages = [{"text": hit["text"]} for hit in hits]

        payload = {
            "model": "nvidia/llama-nemotron-rerank-1b-v2",
            "query": {"text": query},
            "passages": passages,
            "truncate": "END",
        }

        try:
            response = requests.post(
                RERANK_URL,
                headers=self.headers,
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()

            rankings = data.get("rankings", [])
            for ranking in rankings:
                idx = ranking["index"]
                hits[idx]["rerank_score"] = ranking["logit"]

            reranked = sorted(hits, key=lambda x: x.get("rerank_score", 0), reverse=True)
            top = reranked[:top_n]

            log.info(f"Reranked {len(hits)} chunks → kept top {len(top)}")
            return top

        except requests.exceptions.RequestException as e:
            log.error(f"NVIDIA Rerank API error: {e}")
            log.warning("Falling back to vector similarity ordering.")
            return hits[:top_n]
