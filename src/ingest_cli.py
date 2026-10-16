"""Small CLI helper for ingesting files. This can be expanded later into a
proper CLI tool. For now, it exposes a function `cli_ingest` that can be used
by scripts or manual testing."""
from typing import Iterable

from .ingest import ingest


def cli_ingest(paths: Iterable[str], collection_name: str = "default", chunk_size: int = 1000, overlap: int = 200, embedding_fn=None):
    return ingest(paths, collection_name=collection_name, chunk_size=chunk_size, overlap=overlap, embedding_fn=embedding_fn)
