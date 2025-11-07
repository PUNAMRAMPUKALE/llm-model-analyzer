import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "mlProject"

list_of_files = [
    ".github/workflows/.gitkeep",

    # Core app + config
    "app.py",
    "main.py",
    "Dockerfile",
    "requirements.txt",
    "setup.py",
    "params.yaml",
    "schema.yaml",
    "config/config.yaml",
    "README.md",
    "test.py",

    # Research
    "research/trials.ipynb",

    # Templates (for optional UI)
    "templates/index.html",

    # Source packages
    f"src/{project_name}/__init__.py",

    # Entity / schemas
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/schemas.py",

    # Config + constants
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",
    f"src/{project_name}/constants/__init__.py",

    # Utils
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/text.py",

    # Models (embeddings, tokenizers)
    f"src/{project_name}/models/__init__.py",
    f"src/{project_name}/models/embedder.py",

    # Components (metrics)
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/coherence.py",
    f"src/{project_name}/components/redundancy.py",
    f"src/{project_name}/components/coverage.py",
    f"src/{project_name}/components/conciseness.py",
    f"src/{project_name}/components/readability.py",
    f"src/{project_name}/components/hedging.py",
    f"src/{project_name}/components/diversity.py",
    f"src/{project_name}/components/perturb.py",

    # Pipeline + scoring
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/metrics_batch.py",

    # Services
    f"src/{project_name}/services/__init__.py",
    f"src/{project_name}/services/scorer.py",

    # Artifacts / logs / static
    "artifacts/.gitkeep",
    "logs/running_logs.log",
    "static/.gitkeep",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} already exists")
