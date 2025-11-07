from typing import Dict, Any
from ..utils.text import count_hedges, tokens

def hedging_score(text: str) -> Dict[str, Any]:
    n = len(tokens(text)) + 1e-9
    c = count_hedges(text)
    # higher better (fewer hedges)
    hed = max(0.0, 1.0 - (c / (n / 50.0)))  # allow ~1 hedge per 50 tokens
    return {"HED": float(hed), "hedge_count": int(c), "len": int(n)}
