import math
from typing import Dict, Any
from ..utils.text import tokens

def conciseness_score(text: str, target_tokens: int = 220, sigma: int = 120) -> Dict[str, Any]:
    n = len(tokens(text))
    # Gaussian fitness around target length
    con = math.exp(-((n - target_tokens) ** 2) / (2 * (sigma ** 2)))
    return {"CON": float(con), "len": n, "target": target_tokens, "sigma": sigma}
