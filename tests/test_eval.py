import json
import tempfile
import os
from src.eval.harness import run_eval
from src.vector_store import VectorStore
from src.retrieval_agent import RetrievalAgent
from src.eval.baselines import naive_baseline


def simple_embedding(texts):
    out = []
    for t in texts:
        s = sum(ord(c) for c in t)
        out.append([float(s % 100) / 100.0, float((s // 100) % 100) / 100.0, float((s // 10000) % 100) / 100.0])
    return out


def mock_llm(prompt: str) -> str:
    p = (prompt or "").lower()
    if p.startswith("[decide]"):
        # force retrieval for testing except when apple is mentioned
        if "apple" in p:
            return "NO_RETRIEVAL"
        return "RETRIEVE"
    if p.startswith("[synthesize]"):
        return "Synthesized (mock)"
    if p.startswith("[answer_direct]"):
        return "Direct (mock)"
    return ""


def test_eval_harness(tmp_path):
    # create vectorstore and add docs
    vs = VectorStore(collection_name=f"test_eval_{os.getpid()}", embedding_fn=simple_embedding)
    vs.add_texts(["apple is red", "banana is yellow", "transformers are used in nlp"], metadatas=[{"source":"d1"},{"source":"d2"},{"source":"d3"}])

    agent = RetrievalAgent(vs, llm_fn=mock_llm)

    # write a small questions file
    qfile = tmp_path / "qs.jsonl"
    qfile.write_text('\n'.join([
        json.dumps({"id":"q1","question":"What color is an apple?","gold":["red"]}),
        json.dumps({"id":"q2","question":"What fruit is yellow and long?","gold":["banana"]}),
    ]))

    outdir = tmp_path / "out"
    outdir.mkdir()

    res = run_eval(str(qfile), out_dir=str(outdir), vectorstore=vs, agent=agent, baseline_fn=naive_baseline, k=1, dry_run=False)
    assert os.path.exists(res["results_path"]) and os.path.getsize(res["results_path"]) > 0
    assert os.path.exists(res["summary_csv"]) and os.path.exists(res["summary_md"]) 
