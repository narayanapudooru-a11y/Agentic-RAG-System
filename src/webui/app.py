"""Simple Streamlit app for uploading docs and asking questions.

This app uses the local vector store and retrieval agent built in earlier slices.
It is intentionally minimal and keeps the UI logic easy to test.
"""
from __future__ import annotations

import os
import time
from typing import List

import streamlit as st

from src.ingest import ingest
from src.retrieval_agent import RetrievalAgent
from src.vector_store import VectorStore


APP_TITLE = "Agentic RAG Demo"


def build_vector_store(collection_name: str = "ui_collection") -> VectorStore:
    """Create and return a deterministic local vector store for UI use.

    For the real app, replace this with the configured embedding function that uses OpenAI.
    """
    def simple_embedding(texts):
        out = []
        for t in texts:
            s = sum(ord(c) for c in t)
            out.append([float(s % 100) / 100.0, float((s // 100) % 100) / 100.0, float((s // 10000) % 100) / 100.0])
        return out

    return VectorStore(collection_name=collection_name, embedding_fn=simple_embedding)


def render_answer(answer: str, sources: List[dict], latency: float):
    st.subheader("Answer")
    st.write(answer)
    st.caption(f"Latency: {latency:.2f}s")

    if sources:
        st.subheader("Sources")
        for i, s in enumerate(sources, start=1):
            source = s.get("source") or "unknown"
            chunk = s.get("chunk_index")
            text = s.get("text") or ""
            st.markdown(f"**{i}. {source}**")
            if chunk is not None:
                st.caption(f"chunk_index={chunk}")
            st.write(text)


def main():
    st.set_page_config(page_title=APP_TITLE, page_icon="📚")
    st.title(APP_TITLE)

    uploaded = st.file_uploader("Upload a PDF, DOCX, or TXT document", type=["pdf", "docx", "txt"])
    q = st.text_input("Ask a question")

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = build_vector_store()

    if uploaded is not None:
        save_dir = os.path.join(os.getcwd(), "data", "ui_uploads")
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())

        with st.spinner("Ingesting uploaded document..."):
            # Ingest using the same deterministic/simple embedding for the demo UI.
            def simple_embedding(texts):
                out = []
                for t in texts:
                    s = sum(ord(c) for c in t)
                    out.append([float(s % 100) / 100.0, float((s // 100) % 100) / 100.0, float((s // 10000) % 100) / 100.0])
                return out

            ingest([path], collection_name="ui_collection", chunk_size=700, overlap=150, embedding_fn=simple_embedding)

        st.success(f"Uploaded and ingested: {uploaded.name}")

    if q:
        with st.spinner("Searching relevant document chunks..."):
            start = time.perf_counter()
            agent = RetrievalAgent(st.session_state.vectorstore, llm_fn=None)
            result = agent.answer(q, k=3, use_llm=False)
            latency = time.perf_counter() - start

        render_answer(result.get("answer", ""), result.get("sources", []), latency)


if __name__ == "__main__":
    main()
