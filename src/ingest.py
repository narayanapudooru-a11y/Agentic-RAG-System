"""Ingestion utilities: loaders for TXT, PDF, DOCX and an ingest pipeline.

Functions:
- load_txt(path)
- load_pdf(path)
- load_docx(path)
- load_documents(paths) -> list of dicts {id, text, source}
- ingest(paths, collection_name, chunk_size, overlap, embedding_fn) -> stats dict

The ingest function chunks documents, embeds them using the provided embedding_fn
(or the VectorStore's default), and stores them in the Chroma-backed VectorStore
with metadata including source path, original doc id, and chunk index.
"""
from typing import List, Dict, Iterable
import os
import uuid

from .chunking import chunk_text
from .vector_store import VectorStore

try:
    import pypdf
except Exception:
    pypdf = None

try:
    import docx
except Exception:
    docx = None


def load_txt(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def load_pdf(path: str) -> str:
    if pypdf is None:
        raise RuntimeError("pypdf is required to load PDF files")
    reader = pypdf.PdfReader(path)
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:
            # fallback: skip page
            pages.append("")
    return "\n".join(pages)


def load_docx(path: str) -> str:
    if docx is None:
        raise RuntimeError("python-docx is required to load DOCX files")
    doc = docx.Document(path)
    paragraphs = [p.text for p in doc.paragraphs]
    return "\n".join(paragraphs)


def load_document(path: str) -> Dict[str, str]:
    """Load a single document and return a dict with id, text, and source.

    The id is a UUID based on the path to allow deterministic mapping across runs.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    lower = path.lower()
    if lower.endswith(".txt"):
        text = load_txt(path)
    elif lower.endswith(".pdf"):
        text = load_pdf(path)
    elif lower.endswith(".docx"):
        text = load_docx(path)
    else:
        raise RuntimeError(f"Unsupported file type for {path}")

    return {"id": str(uuid.uuid5(uuid.NAMESPACE_URL, os.path.abspath(path))), "text": text, "source": path}


def load_documents(paths: Iterable[str]):
    docs = []
    for p in paths:
        docs.append(load_document(p))
    return docs


def ingest(
    paths: Iterable[str],
    collection_name: str = "default",
    chunk_size: int = 1000,
    overlap: int = 200,
    embedding_fn=None,
):
    """Ingest files into the vector store.

    Returns a stats dict: {num_docs, num_chunks, collection_name}
    """
    docs = load_documents(paths)
    vs = VectorStore(collection_name=collection_name, embedding_fn=embedding_fn)

    texts = []
    metadatas = []
    ids = []
    for d in docs:
        chunks = chunk_text(d["text"], chunk_size=chunk_size, overlap=overlap)
        for i, c in enumerate(chunks):
            texts.append(c)
            metadatas.append({"source": d["source"], "doc_id": d["id"], "chunk_index": i})
            ids.append(f"{d['id']}_{i}")

    if texts:
        vs.add_texts(texts, metadatas=metadatas, ids=ids)

    return {"num_docs": len(docs), "num_chunks": len(texts), "collection_name": collection_name}
