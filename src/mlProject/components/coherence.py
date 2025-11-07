import numpy as np
from typing import Dict, Any, List
from ..utils.text import sentences
from ..models.embedder import Embedder

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0 or b.size == 0:
        return 0.0
    return float(np.dot(a, b))

def coherence_score(text: str, embedder: Embedder) -> Dict[str, Any]:
    sents = sentences(text)
    if len(sents) < 2:
        return {"COH": 0.0, "pairs": []}
    embs = embedder.encode(sents)
    sims: List[float] = []
    for i in range(len(embs)-1):
        sims.append(cosine_sim(embs[i], embs[i+1]))
    return {"COH": float(np.mean(sims)), "pairs": sims}
