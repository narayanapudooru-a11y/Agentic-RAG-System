"""Retrieval agent that decides when to retrieve and synthesizes answers with citations."""
from typing import Callable, Dict, List, Optional

from .vector_store import VectorStore


class RetrievalAgent:
    """Agent that decides whether to retrieve and (optionally) synthesizes answers.

    vectorstore: instance of VectorStore
    llm_fn: optional callable(prompt: str) -> str  (in tests this is a deterministic function)
    retrieve_threshold: float - unused for now but could be used to decide by distance
    """

    def __init__(self, vectorstore: VectorStore, llm_fn: Optional[Callable[[str], str]] = None, retrieve_threshold: float = 0.5):
        self.vs = vectorstore
        self.llm_fn = llm_fn
        self.retrieve_threshold = retrieve_threshold

    def decide_retrieve(self, query: str, k: int = 3) -> bool:
        """Decide whether retrieval is needed.

        If an llm_fn is provided, call it with a short decision prompt. The llm_fn
        should return a string; we treat responses containing 'no' or 'confident'
        as an instruction to skip retrieval. Everything else triggers retrieval.
        If no llm_fn is provided, fall back to a distance-based heuristic using
        the vectorstore: if the top distance is below a threshold, we consider it relevant and retrieve.
        """
        if self.llm_fn is not None:
            prompt = f"[DECIDE] Can you answer the question without retrieving additional documents?\nQuestion: {query}\nReply with a short phrase 'NO_RETRIEVAL' or 'RETRIEVE'."
            resp = self.llm_fn(prompt) or "RETRIEVE"
            rl = resp.strip().lower()
            if "no" in rl or "no_retrieval" in rl or "no retrieval" in rl or "confident" in rl:
                return False
            return True
        # fallback: distance heuristic
        res = self.vs.query(query, k=k)
        distances = res.get("distances", [])
        if not distances:
            return True
        # Chroma distances are smaller == more similar (often cosine distance). Use threshold to decide.
        top_dist = distances[0]
        try:
            top_dist = float(top_dist)
        except Exception:
            # if distance isn't numeric, err on the side of retrieving
            return True
        # If distance is reasonably small (close to 0), we consider retrieval helpful
        return top_dist < self.retrieve_threshold

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        """Retrieve top-k chunks from the vector store and return structured source entries."""
        res = self.vs.query(query, k=k)
        docs = res.get("documents", [])
        metas = res.get("metadatas", [])
        dists = res.get("distances", [])
        out = []
        for text, meta, dist in zip(docs, metas, dists):
            entry = {
                "text": text,
                "source": meta.get("source") if isinstance(meta, dict) else None,
                "doc_id": meta.get("doc_id") if isinstance(meta, dict) else None,
                "chunk_index": meta.get("chunk_index") if isinstance(meta, dict) else None,
                "distance": dist,
            }
            out.append(entry)
        return out

    def answer(self, query: str, k: int = 3, use_llm: bool = True) -> Dict:
        """Answer the query. Returns dict with keys: answer (str), sources (list), retrieved (bool).

        Behavior:
        - If use_llm and llm_fn provided: first ask the llm whether it can answer without retrieval.
          If it says NO_RETRIEVAL, return its response (as 'answer') and retrieved=False.
          Otherwise, retrieve top-k chunks and call llm_fn with a synthesis prompt including the chunks.
        - If no llm_fn is provided: retrieve and return a concatenation of top-k chunks as the 'answer'.
        """
        if use_llm and self.llm_fn is not None:
            # Decision step
            prompt = f"[DECIDE] Can you answer the question without retrieving additional documents?\nQuestion: {query}\nReply with 'NO_RETRIEVAL' or 'RETRIEVE'."
            decide_resp = self.llm_fn(prompt) or "RETRIEVE"
            if any(x in (decide_resp or "").lower() for x in ["no_retrieval", "no retrieval", "no", "confident"]):
                # LLM claims it can answer without retrieval. Ask it to answer directly.
                answer_prompt = f"[ANSWER_DIRECT] Please answer concisely. Question: {query}"
                ans = self.llm_fn(answer_prompt)
                return {"answer": ans, "sources": [], "retrieved": False}
            # Otherwise, retrieve and synthesize
            sources = self.retrieve(query, k=k)
            context = "\n\n".join([f"Source: {s['source']} (chunk {s['chunk_index']})\n{s['text']}" for s in sources])
            synth_prompt = f"[SYNTHESIZE] Use the following sources to answer the question.\n\n{context}\n\nQuestion: {query}\nProvide a short answer and list citations by source and chunk index."
            synth = self.llm_fn(synth_prompt)
            return {"answer": synth, "sources": sources, "retrieved": True}
        # No LLM available: do a retrieve and return concatenated chunks
        sources = self.retrieve(query, k=k)
        answer = "\n\n".join([s["text"] for s in sources])
        return {"answer": answer, "sources": sources, "retrieved": bool(sources)}
