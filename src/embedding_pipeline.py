"""Embedding pipeline helpers."""
from typing import Callable, Iterable, List


def embed_texts(texts: Iterable[str], embedding_fn: Callable[[Iterable[str]], List[List[float]]]):
    """Call the provided embedding function and return a list of vectors."""
    return embedding_fn(list(texts))
