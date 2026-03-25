"""
Streamlit UI: Chat interface for the RAG pipeline.
Streams LLM responses token by token, shows source citations.
"""

import os
import sys
import streamlit as st
from dotenv import load_dotenv
from groq import Groq

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from retriever.embedder import PDFEmbedder
from retriever.vector_store import VectorStore
from retriever.reranker import Reranker

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LLM_MODEL = "llama-3.3-70b-versatile"


@st.cache_resource
def load_pipeline():
    return PDFEmbedder(), VectorStore(), Reranker()


@st.cache_resource
def load_groq():
    return Groq(api_key=GROQ_API_KEY)


def build_prompt(query: str, context_chunks: list) -> str:
    context_text = ""
    for i, chunk in enumerate(context_chunks, 1):
        context_text += f"\n[Source {i}: {chunk['source']}, Page {chunk['page']}]\n{chunk['text']}\n"

    return f"""You are a helpful assistant that answers questions based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information to answer this."
Always cite which source and page your answer comes from.

CONTEXT:
{context_text}

QUESTION: {query}

ANSWER:"""


def query_rag(query, embedder, store, reranker, groq_client):
    query_embedding = embedder.embed([query])[0]
    hits = store.query(query_embedding, top_k=5)

    if not hits:
        return "No documents indexed yet. Drop a PDF into the /docs folder.", []

    top_chunks = reranker.rerank(query, hits, top_n=3)
    prompt = build_prompt(query, top_chunks)

    stream = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
        temperature=0.1,
        max_tokens=512,
    )

    return stream, top_chunks


def main():
    st.set_page_config(page_title="Kafka-RAG", page_icon="⚡", layout="wide")
    st.title("⚡ Kafka-RAG: Real-Time Document Q&A")
    st.caption("Drop PDFs into /docs → indexed automatically via Kafka → ask questions instantly.")

    embedder, store, reranker = load_pipeline()
    groq_client = load_groq()

    with st.sidebar:
        st.header("📚 Indexed Documents")
        sources = store.list_sources()
        if sources:
            for s in sources:
                st.markdown(f"- `{s}`")
        else:
            st.info("No documents indexed yet.\nDrop a PDF into /docs.")

        st.divider()
        st.markdown("**Stack**")
        st.markdown("Kafka · ChromaDB · NVIDIA Nemotron Reranker · Groq Llama3")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander("📄 Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"- **{src['source']}** — Page {src['page']} "
                                    f"(score: {src.get('rerank_score', 0):.3f})")

    if query := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)

        with st.chat_message("assistant"):
            result, top_chunks = query_rag(query, embedder, store, reranker, groq_client)

            if isinstance(result, str):
                st.markdown(result)
                st.session_state.messages.append({
                    "role": "assistant", "content": result, "sources": []
                })
            else:
                response_text = st.write_stream(
                    chunk.choices[0].delta.content or ""
                    for chunk in result
                    if chunk.choices[0].delta.content
                )

                with st.expander("📄 Sources"):
                    for src in top_chunks:
                        st.markdown(f"- **{src['source']}** — Page {src['page']} "
                                    f"(rerank score: {src.get('rerank_score', 0):.3f})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response_text,
                    "sources": top_chunks,
                })


if __name__ == "__main__":
    main()
