FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# System deps (for torch/textstat/nltk)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git wget curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download nltk data at build time to avoid runtime downloads
RUN python - <<'PY'\nimport nltk;nltk.download('punkt')\nPY

COPY . .

EXPOSE 9090
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9090"]

