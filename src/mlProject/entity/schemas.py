from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict, Optional, Any

class MetricWeights(BaseModel):
    COH: float = 0.25
    COV: float = 0.20
    RED: float = 0.15
    CON: float = 0.15
    REA: float = 0.10
    HED: float = 0.05
    DCB: float = 0.10

class MetricsRequest(BaseModel):
    prompt: str
    response: str
    expected_structure: Optional[List[str]] = None
    target_length_tokens: Optional[int] = None
    metric_weights: Optional[MetricWeights] = None

class ResponseWithParams(BaseModel):
    text: str
    params: Dict[str, Any] = Field(default_factory=dict)
    tokens_out: Optional[int] = None
    latency_ms: Optional[int] = None

class BatchMetricsRequest(BaseModel):
    prompt: str
    responses: List[ResponseWithParams]
    config: Dict[str, Any] = Field(default_factory=dict)  # e.g., perturbations

class MetricsResult(BaseModel):
    scores: Dict[str, float]
    details: Dict[str, Any]
    overall_quality: float
    schema_version: str = "metrics.v1.0.0"
    model_versions: Dict[str, str] = Field(default_factory=dict)

class BatchMetricsResult(BaseModel):
    results: List[MetricsResult]
    summary: Dict[str, Any] = Field(default_factory=dict)
