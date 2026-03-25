"""
Producer: Watches /docs folder for new PDFs.
When a file is dropped, publishes an event to Kafka.
Partition key = filename guarantees ordering per document.
"""

import os
import json
import time
import logging
from pathlib import Path
from dotenv import load_dotenv
from confluent_kafka import Producer
from confluent_kafka.admin import AdminClient, NewTopic
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s [PRODUCER] %(message)s")
log = logging.getLogger(__name__)

KAFKA_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
TOPIC = os.getenv("KAFKA_TOPIC", "document-events")
DOCS_DIR = os.getenv("DOCS_WATCH_DIR", "./docs")


def ensure_topic_exists():
    admin = AdminClient({"bootstrap.servers": KAFKA_SERVERS})
    existing = admin.list_topics(timeout=5).topics
    if TOPIC not in existing:
        admin.create_topics([NewTopic(TOPIC, num_partitions=3, replication_factor=1)])
        log.info(f"Created topic: {TOPIC}")


def delivery_report(err, msg):
    if err:
        log.error(f"Delivery failed: {err}")
    else:
        log.info(f"Delivered to {msg.topic()} [{msg.partition()}] @ offset {msg.offset()}")


class DocEventHandler(FileSystemEventHandler):
    def __init__(self, producer: Producer):
        self.producer = producer
        self.seen = set()

    def on_created(self, event):
        if event.is_directory:
            return
        path = Path(event.src_path)
        if path.suffix.lower() != ".pdf":
            return
        if path.name in self.seen:
            return
        self.seen.add(path.name)

        time.sleep(0.5)  # wait for file write to complete

        payload = {
            "filename": path.name,
            "filepath": str(path.resolve()),
            "event_type": "new_document",
            "timestamp": time.time(),
        }

        self.producer.produce(
            topic=TOPIC,
            key=path.name,
            value=json.dumps(payload),
            callback=delivery_report,
        )
        self.producer.flush()
        log.info(f"Published event for: {path.name}")


def main():
    ensure_topic_exists()
    Path(DOCS_DIR).mkdir(exist_ok=True)

    producer = Producer({"bootstrap.servers": KAFKA_SERVERS})
    handler = DocEventHandler(producer)

    observer = Observer()
    observer.schedule(handler, path=DOCS_DIR, recursive=False)
    observer.start()

    log.info(f"Watching {DOCS_DIR} for new PDFs... (Ctrl+C to stop)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
