FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-download the sentence-transformers model
RUN python -c \
    "from sentence_transformers import SentenceTransformer; \
     SentenceTransformer('all-MiniLM-L6-v2')"

EXPOSE 7860

ENV PYTHONPATH=/app
ENV HF_HOME=/app/.cache
ENV MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
ENV OPENAI_BASE_URL=https://api-inference.huggingface.co/v1

CMD ["uvicorn", "src.server:app", \
     "--host", "0.0.0.0", "--port", "7860"]