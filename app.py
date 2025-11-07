from fastapi import FastAPI
from fastapi.responses import ORJSONResponse
from loguru import logger
import yaml
import os

from src.mlProject.models.embedder import Embedder
from src.mlProject.entity.schemas import (
    MetricsRequest, BatchMetricsRequest, MetricsResult, BatchMetricsResult, MetricWeights
)
from src.mlProject.pipeline.metrics_batch import evaluate_single, evaluate_batch

app = FastAPI(default_response_class=ORJSONResponse, title="ML Metrics Service", version="1.0.0")

# Load config
with open(os.path.join("config", "config.yaml"), "r") as f:
    CFG = yaml.safe_load(f)

EMB = Embedder(CFG["ml"]["embedding_model"])
DEFAULT_WEIGHTS = CFG["metrics"]["weights"]
TARGET_LEN = int(CFG["metrics"]["target_length_tokens"])
CON_SIGMA = int(CFG["metrics"]["conciseness_sigma"])
ROB_CFG = CFG.get("robustness", {}).get("perturbations", {"n": 3, "types": ["stopword_drop"]})

@app.get("/health")
def health():
    return {"status": "ok", "embedding_model": CFG["ml"]["embedding_model"]}

@app.post("/metrics", response_model=MetricsResult)
def metrics(req: MetricsRequest):
    weights = DEFAULT_WEIGHTS if req.metric_weights is None else req.metric_weights.model_dump()
    target_len = TARGET_LEN if req.target_length_tokens is None else req.target_length_tokens

    row = evaluate_single(req.prompt, req.response, EMB, target_len, CON_SIGMA, weights)
    result = MetricsResult(
        scores=row["scores"],
        details=row["details"],
        overall_quality=row["overall"],
        model_versions={"embeddings": CFG["ml"]["embedding_model"]}
    )
    return result

@app.post("/metrics/batch", response_model=BatchMetricsResult)
def metrics_batch(req: BatchMetricsRequest):
    weights = DEFAULT_WEIGHTS
    responses = [r.text for r in req.responses]
    rb_cfg = req.config.get("perturbations", ROB_CFG)

    out = evaluate_batch(req.prompt, responses, EMB, TARGET_LEN, CON_SIGMA, weights, rb_cfg)

    results = []
    for row in out["rows"]:
        results.append(MetricsResult(
            scores=row["scores"],
            details=row["details"],
            overall_quality=row["overall"],
            model_versions={"embeddings": CFG["ml"]["embedding_model"]}
        ))
    return BatchMetricsResult(results=results, summary=out["summary"])
