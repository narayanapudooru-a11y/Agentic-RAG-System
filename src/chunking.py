"""Simple character-based text chunking utilities."""
from typing import List


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """Chunk text into pieces of at most chunk_size with given overlap.

    This uses a simple character-based approach that is deterministic and
    easy to test. It avoids splitting in the middle of whitespace-heavy regions
    by trimming each chunk.
    """
    if not text:
        return []
    text = text.strip()
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")

    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    step = chunk_size - overlap
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
        if end >= len(text):
            break
    return chunks
