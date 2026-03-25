"""
Evaluation: RAGAs metrics on the RAG pipeline.
Run after indexing documents to measure answer quality.

Usage:
    python scripts/evaluate.py
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from groq import Groq

from retriever.embedder import PDFEmbedder
from retriever.vector_store import VectorStore
from retriever.reranker import Reranker

load_dotenv()

TEST_QUESTIONS = [
    "What is the main topic of the document?",
    "What are the key findings or conclusions?",
    "What methodology is described?",
]


def run_eval():
    embedder = PDFEmbedder()
    store = VectorStore()
    reranker = Reranker()
    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    rows = []
    for question in TEST_QUESTIONS:
        query_emb = embedder.embed([question])[0]
        hits = store.query(query_emb, top_k=5)
        if not hits:
            print(f"No docs for: {question}")
            continue

        top_chunks = reranker.rerank(question, hits, top_n=3)
        contexts = [c["text"] for c in top_chunks]
        context_text = "\n\n".join(contexts)

        prompt = f"Answer based only on this context:\n{context_text}\n\nQuestion: {question}\nAnswer:"
        response = groq_client.chat.completions.create(
            model="llama3-8b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=256,
        )
        answer = response.choices[0].message.content

        rows.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": "",
        })
        print(f"Q: {question}\nA: {answer[:100]}...\n")

    if not rows:
        print("No results to evaluate.")
        return

    dataset = Dataset.from_list(rows)
    result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision])

    print("\n── RAGAs Evaluation Results ──")
    print(f"Faithfulness:      {result['faithfulness']:.3f}")
    print(f"Answer Relevancy:  {result['answer_relevancy']:.3f}")
    print(f"Context Precision: {result['context_precision']:.3f}")
    print("──────────────────────────────")


if __name__ == "__main__":
    run_eval()
