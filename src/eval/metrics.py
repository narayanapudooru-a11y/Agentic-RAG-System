from typing import Dict, Iterable, List
import re


def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^a-z0-9 ]", "", s)
    return s


def exact_match(pred: str, golds: Iterable[str]) -> int:
    p = normalize_text(pred)
    for g in golds:
        if p == normalize_text(g):
            return 1
    return 0


def f1_score(pred: str, gold: str) -> float:
    p_tokens = normalize_text(pred).split()
    g_tokens = normalize_text(gold).split()
    if not p_tokens and not g_tokens:
        return 1.0
    if not p_tokens or not g_tokens:
        return 0.0
    common = set(p_tokens) & set(g_tokens)
    if not common:
        return 0.0
    prec = len(common) / len(p_tokens)
    rec = len(common) / len(g_tokens)
    if prec + rec == 0:
        return 0.0
    return 2 * (prec * rec) / (prec + rec)


def score_answer(pred: str, golds: Iterable[str]) -> Dict:
    """Compute exact match and max-F1 across gold answers."""
    golds = list(golds or [])
    if not golds:
        return {"exact_match": 0.0, "f1": 0.0}
    em = exact_match(pred, golds)
    f1s = [f1_score(pred, g) for g in golds]
    return {"exact_match": float(em), "f1": max(f1s) if f1s else 0.0}


def aggregate_scores(scores: Iterable[Dict]) -> Dict:
    scores = list(scores)
    if not scores:
        return {"exact_match": 0.0, "f1": 0.0}
    mean_em = sum(s.get("exact_match", 0.0) for s in scores) / len(scores)
    mean_f1 = sum(s.get("f1", 0.0) for s in scores) / len(scores)
    return {"exact_match": mean_em, "f1": mean_f1}
