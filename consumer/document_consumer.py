"""
Consumer: Listens to Kafka topic, processes new PDFs into ChromaDB.
Runs continuously. Manual commit ensures no message is lost on failure.
"""

import os
import json
import logging
from dotenv import load_dotenv
from confluent_kafka import Consumer, KafkaError

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.embedder import PDFEmbedder
from retriever.vector_store import VectorStore

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [CONSUMER] %(message)s")
log = logging.getLogger(__name__)

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "document-events")
GROUP_ID = "rag-indexer-group"


def build_consumer() -> Consumer:
    return Consumer({
        "bootstrap.servers": KAFKA_SERVERS,
        "group.id": GROUP_ID,
        "auto.offset.reset": "earliest",
        "enable.auto.commit": False,  # manual commit only after success
    })


def process_message(payload: dict, embedder: PDFEmbedder, store: VectorStore):
    filepath = payload["filepath"]
    filename = payload["filename"]

    if not os.path.exists(filepath):
        log.warning(f"File not found: {filepath} — skipping.")
        return

    log.info(f"Processing: {filename}")
    chunks, embeddings = embedder.process_pdf(filepath, filename)

    if not chunks:
        log.warning(f"No chunks extracted from {filename}")
        return

    store.upsert(chunks, embeddings)
    log.info(f"Indexed {filename} ({len(chunks)} chunks)")


def main():
    embedder = PDFEmbedder()
    store = VectorStore()
    consumer = build_consumer()
    consumer.subscribe([TOPIC])

    log.info(f"Subscribed to {TOPIC}. Waiting for documents...")

    try:
        while True:
            msg = consumer.poll(timeout=1.0)

            if msg is None:
                continue
            if msg.error():
                if msg.error().code() == KafkaError._PARTITION_EOF:
                    continue
                log.error(f"Kafka error: {msg.error()}")
                continue

            try:
                payload = json.loads(msg.value().decode("utf-8"))
                process_message(payload, embedder, store)
                consumer.commit(msg)  # commit only after successful indexing
            except Exception as e:
                log.error(f"Failed to process message: {e}", exc_info=True)

    except KeyboardInterrupt:
        log.info("Shutting down consumer.")
    finally:
        consumer.close()


if __name__ == "__main__":
    main()
