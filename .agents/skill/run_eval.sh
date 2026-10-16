#!/usr/bin/env bash
set -euo pipefail

# Simple runner script for the eval harness. Intended for local/manual runs.
# Usage:
# bash .agents/skill/run_eval.sh --questions data/eval/questions.jsonl --out out/eval

QUESTIONS="${1:-data/eval/questions.jsonl}"
OUT_DIR="${2:-out/eval}"

python - <<'PY'
import os
from src.eval.harness import run_eval
from src.vector_store import VectorStore
from src.retrieval_agent import RetrievalAgent

# For local demo we use deterministic embeddings and a mock LLM

def simple_embedding(texts):
    out = []
    for t in texts:
        s = sum(ord(c) for c in t)
        out.append([float(s % 100) / 100.0, float((s // 100) % 100) / 100.0, float((s // 10000) % 100) / 100.0])
    return out

# mock llm
def mock_llm(prompt: str) -> str:
    p = (prompt or "").lower()
    if p.startswith("[decide]"):
        return "RETRIEVE"
    if p.startswith("[synthesize]"):
        return "Synthesized answer (mock)"
    return ""

os.makedirs('