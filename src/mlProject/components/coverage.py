from typing import Dict, Any, Set
from ..utils.text import tokens
from ..models.embedder import Embedder
import numpy as np

def coverage_score(prompt: str, response: str, embedder: Embedder, top_k: int = 8) -> Dict[str, Any]:
    # crude keywording: take top_k frequent non-trivial prompt tokens
    stop = {"the","a","an","to","and","or","of","in","on","for","is","are","be","with","that","this","it","as","by"}
    p_toks = [t for t in tokens(prompt) if t not in stop]
    if not p_toks:
        return {"COV": 0.0, "keywords": [], "hits": []}

    # choose unique keywords by frequency
    freq = {}
    for t in p_toks:
        freq[t] = freq.get(t, 0) + 1
    keywords = sorted(freq, key=freq.get, reverse=True)[:top_k]

    r_toks_set: Set[str] = set(tokens(response))
    lexical_hits = [k for k in keywords if k in r_toks_set]
    lexical_rate = len(lexical_hits) / max(1, len(keywords))

    # semantic match (embed and cosine to any response sentence chunk)
    # embed keywords as phrases; response as sentence-level chunks
    # to keep it light, just embed keywords vs full response
    kw_embs = embedder.encode(keywords)
    resp_emb = embedder.encode([response]).squeeze(0)
    if resp_emb.size == 0 or kw_embs.size == 0:
        sem_rate = 0.0
    else:
        sims = kw_embs @ resp_emb
        sem_rate = float((sims > 0.4).sum()) / max(1, len(keywords))

    cov = 0.5 * lexical_rate + 0.5 * sem_rate
    return {"COV": float(cov), "keywords": keywords, "lexical_hits": lexical_hits, "sem_rate": sem_rate}
