import time
from typing import Dict


def naive_baseline(query: str, vectorstore, k: int = 1) -> Dict:
    """Baseline that returns the top-1 retrieved chunk verbatim as the answer."""
    t0 = time.perf_counter()
    res = vectorstore.query(query, k=k)
    t1 = time.perf_counter()
    docs = res.get("documents", [])
    metas = res.get("metadatas", [])
    sources = []
    for d, m in zip(docs, metas):
        sources.append({"text": d, "meta": m})
    answer = docs[0] if docs else ""
    return {"answer": answer, "sources": sources, "latency": t1 - t0}
