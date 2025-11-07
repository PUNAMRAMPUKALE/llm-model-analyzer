from typing import Dict, Any
from ..utils.text import tokens, unique_ratio_ngrams, shannon_entropy
import math

def redundancy_score(text: str) -> Dict[str, Any]:
    toks = tokens(text)
    if not toks:
        return {"RED": 1.0, "entropy": 0.0, "unique_bi": 1.0, "len": 0}

    entropy = shannon_entropy(toks)
    V = len(set(toks)) + 1e-9
    entropy_norm = entropy / math.log(V + 1e-9)  # 0..1
    unique_bi = unique_ratio_ngrams(toks, 2)

    # Combine (higher better = less redundant)
    red = 0.5 * entropy_norm + 0.5 * unique_bi
    return {"RED": float(red), "entropy": float(entropy), "unique_bi": float(unique_bi), "len": len(toks)}
