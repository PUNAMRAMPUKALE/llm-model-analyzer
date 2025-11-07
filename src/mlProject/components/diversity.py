from typing import List, Dict, Any
from ..utils.text import tokens, ngrams
import math

def distinct_n(texts: List[str], n: int = 2) -> float:
    all_ngrams = []
    for t in texts:
        all_ngrams += ngrams(tokens(t), n)
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)

def diversity_coherence_balance(diversity: float, mean_coherence: float) -> float:
    # harmonic mean to reward balance
    eps=1e-6
    if diversity<=0 or mean_coherence<=0:
        return 0.0
    return 2.0 * (diversity * mean_coherence) / (diversity + mean_coherence + eps)

def self_bleu_placeholder(_: List[str]) -> float:
    # Keep simple for now; Distinct-n is robust and fast for demo
    return 0.0

def dcb_score(responses: List[str], mean_coh: float) -> Dict[str, Any]:
    d2 = distinct_n(responses, 2)
    score = diversity_coherence_balance(d2, mean_coh)
    return {"DCB": float(score), "distinct2": float(d2), "mean_coh": float(mean_coh)}
