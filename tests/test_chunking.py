import os
from src.chunking import chunk_text


def test_chunking_small():
    s = "short text"
    chunks = chunk_text(s, chunk_size=100, overlap=10)
    assert chunks == ["short text"]


def test_chunking_exact():
    s = "a" * 300
    chunks = chunk_text(s, chunk_size=100, overlap=10)
    # length 300 -> step 90 -> chunks: positions 0-100,90-190,180-280,270-300 => 4 chunks
    assert len(chunks) == 4


def test_chunking_invalid():
    try:
        chunk_text("x", chunk_size=0)
        assert False, "expected ValueError"
    except ValueError:
        pass
