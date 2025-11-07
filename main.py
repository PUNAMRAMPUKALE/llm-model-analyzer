import yaml
from src.mlProject.models.embedder import Embedder
from src.mlProject.pipeline.metrics_batch import evaluate_single

CFG = yaml.safe_load(open("config/config.yaml"))
EMB = Embedder(CFG["ml"]["embedding_model"])

prompt = "Explain how temperature and top_p affect LLM output, with examples."
resp = "Temperature controls randomness. Higher values increase creativity but may reduce coherence. Top_p limits sampling to a nucleus of probable tokens. Use low temp for accuracy, higher for brainstorming."

row = evaluate_single(prompt, resp, EMB, CFG["metrics"]["target_length_tokens"], CFG["metrics"]["conciseness_sigma"], CFG["metrics"]["weights"])
print("Scores:", row["scores"])
print("Overall:", row["overall"])
