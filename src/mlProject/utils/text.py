import re
import math
from collections import Counter
from typing import List, Tuple
import nltk

# Ensure punkt downloaded (safe to call; checks cache)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

_sentence_splitter = nltk.data.load("tokenizers/punkt/english.pickle")

HEDGES = {"might","may","could","perhaps","probably","possibly","seems","appears","likely","unlikely","i think","i believe"}

def sentences(text: str) -> List[str]:
    sents = _sentence_splitter.tokenize(text.strip())
    return [s.strip() for s in sents if s.strip()]

def tokens(text: str) -> List[str]:
    # Simple whitespace/punct split; robust and fast
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())

def ngrams(seq: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(seq[i:i+n]) for i in range(len(seq)-n+1)]

def shannon_entropy(seq: List[str]) -> float:
    if not seq:
        return 0.0
    counts = Counter(seq)
    total = len(seq)
    return -sum((c/total)*math.log(c/total + 1e-12) for c in counts.values())

def unique_ratio_ngrams(seq: List[str], n: int) -> float:
    ns = ngrams(seq, n)
    if not ns:
        return 1.0
    return len(set(ns)) / len(ns)

def count_hedges(text: str) -> int:
    t = text.lower()
    return sum(t.count(h) for h in HEDGES)

def has_bullets(text: str) -> bool:
    return bool(re.search(r"^\s*[-*•]\s+", text, flags=re.MULTILINE))

def has_numbered_list(text: str) -> bool:
    return bool(re.search(r"^\s*\d+[\.)]\s+", text, flags=re.MULTILINE))

def has_code_fence(text: str) -> bool:
    return "```" in text or re.search(r"`[^`]+`", text) is not None

def heading_count(text: str) -> int:
    return len(re.findall(r"^\s*#{1,6}\s+\S+", text, flags=re.MULTILINE))
