"""
Embedder: PDF parsing, chunking, and embedding.
Written from primitives — no LangChain dependency.
"""

import fitz  # PyMuPDF
import logging
from dataclasses import dataclass
from typing import List
from sentence_transformers import SentenceTransformer

log = logging.getLogger(__name__)

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50


@dataclass
class Chunk:
    text: str
    source: str
    page: int
    chunk_index: int
    chunk_id: str


class PDFEmbedder:
    def __init__(self):
        log.info(f"Loading embedding model: {EMBED_MODEL}")
        self.model = SentenceTransformer(EMBED_MODEL)
        log.info("Embedding model ready.")

    def parse_pdf(self, filepath: str) -> List[dict]:
        doc = fitz.open(filepath)
        pages = []
        for page_num, page in enumerate(doc):
            text = page.get_text("text").strip()
            if text:
                pages.append({"text": text, "page": page_num + 1})
        doc.close()
        log.info(f"Parsed {len(pages)} pages from {filepath}")
        return pages

    def chunk_text(self, text: str, source: str, page: int) -> List[Chunk]:
        separators = ["\n\n", "\n", ". ", " ", ""]
        chunks = []
        chunk_index = 0

        def split(text, separators):
            if not text.strip():
                return []
            if len(text) <= CHUNK_SIZE:
                return [text.strip()]

            sep = separators[0] if separators else ""
            parts = text.split(sep) if sep else list(text)

            results = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= CHUNK_SIZE:
                    current = candidate
                else:
                    if current:
                        results.append(current.strip())
                    overlap_text = current[-CHUNK_OVERLAP:] if len(current) > CHUNK_OVERLAP else current
                    current = overlap_text + sep + part if overlap_text else part

                    if len(current) > CHUNK_SIZE and len(separators) > 1:
                        sub = split(current, separators[1:])
                        results.extend(sub[:-1])
                        current = sub[-1] if sub else ""
            if current:
                results.append(current.strip())
            return results

        raw_chunks = split(text, separators)
        for raw in raw_chunks:
            if raw:
                chunk_id = f"{source}_p{page}_c{chunk_index}"
                chunks.append(Chunk(
                    text=raw,
                    source=source,
                    page=page,
                    chunk_index=chunk_index,
                    chunk_id=chunk_id,
                ))
                chunk_index += 1
        return chunks

    def embed(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts, batch_size=32, show_progress_bar=False).tolist()

    def process_pdf(self, filepath: str, filename: str):
        pages = self.parse_pdf(filepath)
        all_chunks = []
        for page_data in pages:
            chunks = self.chunk_text(page_data["text"], filename, page_data["page"])
            all_chunks.extend(chunks)

        log.info(f"Generated {len(all_chunks)} chunks from {filename}")
        texts = [c.text for c in all_chunks]
        embeddings = self.embed(texts)
        log.info(f"Embedded {len(embeddings)} chunks.")
        return all_chunks, embeddings
