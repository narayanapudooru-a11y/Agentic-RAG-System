import uuid

from src.vector_store import VectorStore
from src.retrieval_agent import RetrievalAgent


def simple_embedding(texts):
    # same deterministic embedding used elsewhere
    out = []
    for t in texts:
        s = sum(ord(c) for c in t)
        out.append([float(s % 100) / 100.0, float((s // 100) % 100) / 100.0, float((s // 10000) % 100) / 100.0])
    return out


def mock_llm(prompt: str) -> str:
    p = (prompt or "").lower()
    if p.startswith("[decide]"):
        # Simple rule: if the question contains 'apple' say no retrieval needed
        if "apple" in p:
            return "NO_RETRIEVAL: confident"
        return "RETRIEVE"
    if p.startswith("[answer_direct]"):
        return "Direct answer from LLM"
    if p.startswith("[synthesize]"):
        return "Synthesized answer based on sources [source1]"
    return ""


def test_retrieval_agent_distance_decision():
    vs = VectorStore(collection_name=f"test_ra_{uuid.uuid4()}", embedding_fn=simple_embedding)
    texts = ["apple is red", "banana is yellow", "grape is purple"]
    metas = [{"source": "t1"}, {"source": "t2"}, {"source": "t3"}]
    ids = vs.add_texts(texts, metadatas=metas)

    agent = RetrievalAgent(vs, llm_fn=None, retrieve_threshold=1.0)
    res = agent.answer("apple", k=2, use_llm=False)
    assert res["retrieved"] is True
    assert res["sources"]
    assert any("apple" in s["text"] for s in res["sources"])


def test_retrieval_agent_llm_no_retrieve():
    vs = VectorStore(collection_name=f"test_ra2_{uuid.uuid4()}", embedding_fn=simple_embedding)
    texts = ["apple is red", "banana is yellow"]
    metas = [{"source": "t1"}, {"source": "t2"}]
    vs.add_texts(texts, metadatas=metas)

    agent = RetrievalAgent(vs, llm_fn=mock_llm)
    res = agent.answer("apple", k=2, use_llm=True)
    assert res["retrieved"] is False
    assert res["answer"] == "Direct answer from LLM"


def test_retrieval_agent_llm_with_synthesis():
    vs = VectorStore(collection_name=f"test_ra3_{uuid.uuid4()}", embedding_fn=simple_embedding)
    texts = ["deep learning overview", "transformers are state of the art"]
    metas = [{"source": "d1"}, {"source": "d2"}]
    vs.add_texts(texts, metadatas=metas)

    agent = RetrievalAgent(vs, llm_fn=mock_llm)
    res = agent.answer("What is transformer?", k=2, use_llm=True)
    assert res["retrieved"] is True
    assert "Synthesized" in res["answer"] or "Synthesized" in res["answer"].lower()
