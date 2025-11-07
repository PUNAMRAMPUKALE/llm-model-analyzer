from typing import Dict, Any, List
import numpy as np

def zscore(values: List[float]) -> List[float]:
    arr = np.array(values, dtype=np.float32)
    if arr.size == 0:
        return []
    mu, sd = float(arr.mean()), float(arr.std() + 1e-6)
    return list((arr - mu) / sd)

def combine_scores(scores: Dict[str, float], weights: Dict[str, float]) -> float:
    # Weighted sum (assumes each metric roughly ~0..1; batch-wise normalization handled outside if desired)
    total = 0.0
    for k, w in weights.items():
        total += w * float(scores.get(k, 0.0))
    return float(total)
