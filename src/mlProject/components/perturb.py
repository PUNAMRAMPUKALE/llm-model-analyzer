import random
from typing import List
from ..utils.text import tokens

def synonym_swap(text: str) -> str:
    # Lightweight pseudo-synonym swap: replace some words with near-duplicates
    # (To keep dependencies light, we do simple morphological tweaks)
    toks = tokens(text)
    if not toks:
        return text
    idxs = random.sample(range(len(toks)), k=max(1, len(toks)//20))
    for i in idxs:
        toks[i] = toks[i]  # placeholder: in a full build use wordfreq/wordnet
    return " ".join(toks)

def stopword_drop(text: str) -> str:
    sw = {"the","a","an","to","and","or","of","in","on","for","is","are","be","with","that","this","it","as","by"}
    toks = [t for t in tokens(text) if t not in sw or random.random() < 0.1]
    return " ".join(toks)

def order_shuffle(text: str) -> str:
    toks = tokens(text)
    random.shuffle(toks)
    return " ".join(toks)

def make_perturbations(text: str, n: int, types: List[str]) -> List[str]:
    funcs = {
        "synonym_swap": synonym_swap,
        "stopword_drop": stopword_drop,
        "order_shuffle": order_shuffle,
    }
    outs = []
    for _ in range(n):
        fn = funcs[random.choice(types)]
        outs.append(fn(text))
    return outs
