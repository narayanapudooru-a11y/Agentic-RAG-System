"""Vector store wrapper using Chroma with pluggable embedding function.

This module provides a minimal VectorStore class that can add texts (with
optional metadatas and ids) and query for nearest neighbors. By default it
uses the OpenAI embeddings API (via the openai package) if no embedding_fn
is provided. For unit tests and CI, pass a deterministic embedding_fn to
avoid network calls.
"""
from typing import Callable, Iterable, List, Optional
import uuid
import os

import chromadb
from chromadb.utils import embedding_functions

try:
    import openai
except Exception:
    openai = None


class VectorStore:
    """Simple wrapper around chromadb collections.

    Parameters
    - collection_name: name of the Chroma collection (created if missing)
    - embedding_fn: optional callable: str -> List[float] or List[List[float]]
      If None, uses OpenAI text-embedding-3-small via openai.Embedding.create.
    """

    def __init__(
        self,
        collection_name: str = "default",
        embedding_fn: Optional[Callable[[Iterable[str]], List[List[float]]]] = None,
    ) -> None:
        self.client = chromadb.Client()
        # get or create collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)

        if embedding_fn is None:
            # wrap openai embeddings
            if openai is None:
                raise RuntimeError("openai package is required for default embedding_fn")
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                openai.api_key = openai_api_key

            def _openai_embed(texts: Iterable[str]):
                # Use OpenAI embeddings API (text-embedding-3-small)
                # Returns a list of embedding vectors (list[float])
                resp = openai.Embedding.create(model="text-embedding-3-small", input=list(texts))
                return [e["embedding"] for e in resp["data"]]

            self.embedding_fn = _openai_embed
        else:
            self.embedding_fn = embedding_fn

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Embed and add texts to the collection. Returns the ids used."""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
        if metadatas is None:
            metadatas = [{} for _ in texts]

        embeddings = self.embedding_fn(texts)
        # Ensure embeddings are lists (not numpy arrays)
        embeddings = [list(map(float, e)) for e in embeddings]

        self.collection.add(
            ids=ids,
            documents=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )
        return ids

    def query(self, query: str, k: int = 3):
        """Query the collection, returning the raw Chroma result for inspection.

        Result keys include: ids, documents, metadatas, distances.
        """
        q_emb = self.embedding_fn([query])
        q_emb = list(map(float, q_emb[0]))
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["ids", "documents", "metadatas", "distances"],
        )
        # Chroma returns each field as a list of lists (one per query). We only
        # issued one query, so extract the first element for easier use.
        return {
            "ids": res.get("ids", [[]])[0],
            "documents": res.get("documents", [[]])[0],
            "metadatas": res.get("metadatas", [[]])[0],
            "distances": res.get("distances", [[]])[0],
        }
