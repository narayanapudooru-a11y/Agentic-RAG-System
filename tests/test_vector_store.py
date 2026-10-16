import tempfile

from src.vector_store import VectorStore


def simple_embedding(texts):
    """Deterministic toy embedding for tests. Maps text -> 3-d vector derived from ord sums."""
    out = []
    for t in texts:
        s = sum(ord(c) for c in t)
        out.append([float(s % 100) / 100.0, float((s // 100) % 100) / 100.0, float((s // 10000) % 100) / 100.0])
    return out


def test_add_and_query():
    vs = VectorStore(collection_name="test_collection", embedding_fn=simple_embedding)

    texts = ["apple", "banana", "apricot"]
    metas = [{"source": "a"}, {"source": "b"}, {"source": "c"}]
    ids = vs.add_texts(texts, metadatas=metas)
    assert len(ids) == 3

    # Query for 'apple' and expect 'apple' to be the top result
    res = vs.query("apple", k=2)
    assert "documents" in res
    assert res["documents"]
    assert res["documents"][0] == "apple"
    assert res["metadatas"][0]["source"] == "a"
