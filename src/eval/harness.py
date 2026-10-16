import json
import os
import time
from typing import Callable, Dict, Iterable, Optional

from ..vector_store import VectorStore
from ..retrieval_agent import RetrievalAgent
from .baselines import naive_baseline
from .metrics import score_answer, aggregate_scores


def run_eval(
    question_file: str,
    out_dir: str,
    collection_name: Optional[str] = None,
    vectorstore: Optional[VectorStore] = None,
    agent: Optional[RetrievalAgent] = None,
    baseline_fn: Optional[Callable] = None,
    embedding_fn: Optional[Callable] = None,
    llm_fn: Optional[Callable] = None,
    k: int = 3,
    dry_run: bool = True,
):
    """Run evaluation comparing an agentic system against a baseline.

    Parameters:
    - question_file: path to JSONL file with questions: {id, question, gold: [answers]}
    - out_dir: directory to write results (JSONL per-question and summary CSV/MD)
    - collection_name: name for vectorstore if vectorstore not provided
    - vectorstore: optional VectorStore instance
    - agent: optional RetrievalAgent instance
    - baseline_fn: function(query, vectorstore, k) -> dict(answer, sources, latency)
    - embedding_fn: optional embedding function for creating a vectorstore
    - llm_fn: optional LLM function for the agent
    - k: number of retrieved chunks
    - dry_run: if True use mock LLM/embeddings when not provided
    """
    os.makedirs(out_dir, exist_ok=True)

    # Prepare vectorstore and agent defaults if not provided
    if vectorstore is None:
        if collection_name is None:
            collection_name = "eval_collection"
        vectorstore = VectorStore(collection_name=collection_name, embedding_fn=embedding_fn)
    if agent is None:
        agent = RetrievalAgent(vectorstore, llm_fn=llm_fn)
    if baseline_fn is None:
        baseline_fn = naive_baseline

    # read questions
    questions = []
    with open(question_file, "r", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            questions.append(json.loads(line))

    results_path = os.path.join(out_dir, "results.jsonl")
    summary = []

    for q in questions:
        qid = q.get("id")
        question = q.get("question")
        golds = q.get("gold", [])

        # baseline
        t0 = time.perf_counter()
        baseline_res = baseline_fn(question, vectorstore, k=1)
        t1 = time.perf_counter()
        baseline_latency = t1 - t0
        baseline_answer = baseline_res.get("answer")
        baseline_sources = baseline_res.get("sources")

        # agent
        t0 = time.perf_counter()
        agent_res = agent.answer(question, k=k, use_llm=not dry_run)
        t1 = time.perf_counter()
        agent_latency = t1 - t0

        agent_answer = agent_res.get("answer")
        agent_sources = agent_res.get("sources")

        # score
        baseline_metrics = score_answer(baseline_answer, golds)
        agent_metrics = score_answer(agent_answer, golds)

        entry = {
            "id": qid,
            "question": question,
            "gold": golds,
            "baseline_answer": baseline_answer,
            "baseline_sources": baseline_sources,
            "baseline_latency": baseline_latency,
            "baseline_metrics": baseline_metrics,
            "agent_answer": agent_answer,
            "agent_sources": agent_sources,
            "agent_latency": agent_latency,
            "agent_metrics": agent_metrics,
        }

        with open(results_path, "a", encoding="utf-8") as outfh:
            outfh.write(json.dumps(entry) + "\n")

        summary.append(entry)

    # aggregate
    agg = {
        "baseline": aggregate_scores([s["baseline_metrics"] for s in summary]),
        "agent": aggregate_scores([s["agent_metrics"] for s in summary]),
        "baseline_mean_latency": sum(s["baseline_latency"] for s in summary) / max(1, len(summary)),
        "agent_mean_latency": sum(s["agent_latency"] for s in summary) / max(1, len(summary)),
    }

    # write summary files
    csv_path = os.path.join(out_dir, "summary.csv")
    md_path = os.path.join(out_dir, "summary.md")

    # simple CSV
    import csv

    with open(csv_path, "w", newline="", encoding="utf-8") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["system", "mean_exact_match", "mean_f1"])
        writer.writerow(["baseline", agg["baseline"].get("exact_match", 0), agg["baseline"].get("f1", 0)])
        writer.writerow(["agent", agg["agent"].get("exact_match", 0), agg["agent"].get("f1", 0)])

    with open(md_path, "w", encoding="utf-8") as mdf:
        mdf.write("# Evaluation Summary\n\n")
        mdf.write("| system | mean_exact_match | mean_f1 | mean_latency(s) |\n")
        mdf.write("|---|---:|---:|---:|\n")
        mdf.write(f"| baseline | {agg['baseline'].get('exact_match',0):.3f} | {agg['baseline'].get('f1',0):.3f} | {agg['baseline_mean_latency']:.4f} |\n")
        mdf.write(f"| agent | {agg['agent'].get('exact_match',0):.3f} | {agg['agent'].get('f1',0):.3f} | {agg['agent_mean_latency']:.4f} |\n")

    return {"results_path": results_path, "summary_csv": csv_path, "summary_md": md_path}
