from typing import List, Dict, Any
import numpy as np
from ..models.embedder import Embedder
from ..components.coherence import coherence_score
from ..components.redundancy import redundancy_score
from ..components.coverage import coverage_score
from ..components.conciseness import conciseness_score
from ..components.readability import readability_score
from ..components.hedging import hedging_score
from ..components.diversity import dcb_score
from ..components.perturb import make_perturbations
from ..services.scorer import combine_scores

def evaluate_single(prompt: str,
                    response: str,
                    embedder: Embedder,
                    target_len: int,
                    conc_sigma: int,
                    weights: Dict[str, float]) -> Dict[str, Any]:

    coh = coherence_score(response, embedder)              # COH
    red = redundancy_score(response)                       # RED
    cov = coverage_score(prompt, response, embedder)       # COV
    con = conciseness_score(response, target_len, conc_sigma)  # CON
    rea = readability_score(response)                      # REA
    hed = hedging_score(response)                          # HED

    scores = {
        "COH": coh["COH"], "RED": red["RED"], "COV": cov["COV"],
        "CON": con["CON"], "REA": rea["REA"], "HED": hed["HED"]
    }
    overall = combine_scores(scores, weights)
    details = {
        "adjacent_cosines": coh.get("pairs", []),
        "entropy": red.get("entropy"), "unique_bi": red.get("unique_bi"),
        "keywords": cov.get("keywords"), "lexical_hits": cov.get("lexical_hits"), "sem_rate": cov.get("sem_rate"),
        "len": con.get("len"), "target": con.get("target"),
        "flesch": rea.get("flesch"), "grade": rea.get("grade"),
        "hedge_count": hed.get("hedge_count")
    }
    return {"scores": scores, "overall": overall, "details": details}

def evaluate_batch(prompt: str,
                   responses: List[str],
                   embedder: Embedder,
                   target_len: int,
                   conc_sigma: int,
                   weights: Dict[str, float],
                   robustness_cfg: Dict[str, Any]) -> Dict[str, Any]:
    # Base evaluation
    rows = [evaluate_single(prompt, r, embedder, target_len, conc_sigma, weights) for r in responses]

    # Robustness (SRS): stddev of overall when perturbing response text
    n = int(robustness_cfg.get("n", 3))
    types = robustness_cfg.get("types", ["stopword_drop"])

    srs_list = []
    for i, r in enumerate(responses):
        pert = make_perturbations(r, n, types)
        ovs = [evaluate_single(prompt, p, embedder, target_len, conc_sigma, weights)["overall"] for p in pert]
        std = float(np.std(ovs)) if ovs else 0.0
        srs = float(max(0.0, 1.0 - std))  # higher better (stable)
        srs_list.append(srs)
        rows[i]["scores"]["SRS"] = srs

    # DCB (diversity-coherence balance) across batch
    mean_coh = float(np.mean([row["scores"]["COH"] for row in rows])) if rows else 0.0
    dcb = dcb_score(responses, mean_coh)
    for row in rows:
        row["scores"]["DCB"] = dcb["DCB"]

    # Recompute overall with DCB included if present in weights
    for row in rows:
        row["overall"] = combine_scores(row["scores"], weights)

    return {
        "rows": rows,
        "summary": {
            "mean_coherence": mean_coh,
            "dcb": dcb,
            "mean_srs": float(np.mean(srs_list)) if srs_list else 0.0
        }
    }
