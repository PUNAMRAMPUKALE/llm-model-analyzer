from functools import lru_cache
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer

@lru_cache(maxsize=1)
def _load_model(name: str) -> SentenceTransformer:
    return SentenceTransformer(name)

class Embedder:
    def __init__(self, model_name: str):
        self.model = _load_model(model_name)

    def encode(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        embs = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embs, dtype=np.float32)