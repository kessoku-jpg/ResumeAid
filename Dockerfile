# =======================================================================
# ResumeAid Backend — Dockerfile (Hugging Face Spaces / Cloud Docker)
# =======================================================================
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.cache/huggingface \
    PORT=7860

WORKDIR /app

# Install CPU-only PyTorch first to avoid pulling ~2.5 GB of unused CUDA packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install backend Python dependencies & spaCy English model
COPY backend/requirements.txt /app/backend/requirements.txt
RUN pip install --no-cache-dir -r /app/backend/requirements.txt && \
    python -m spacy download en_core_web_sm

# Pre-download the Sentence-BERT model into the image so cold starts work offline & fast
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Copy application files (.dockerignore excludes /onet, /evaluation, /logs, and venvs)
COPY index.html app.js style.css /app/
COPY fonts/ /app/fonts/
COPY backend/ /app/backend/

# Warm/verify all O*NET embedding caches (.pt) at build time so startup is instant
RUN python -c "import sys; sys.path.insert(0, '/app/backend'); import matching_engine as me; me._get_skill_tree_embeddings()"

# Ensure cache and session log directories are writable by non-root user (HF Spaces uses uid 1000)
RUN mkdir -p /app/logs /app/backend/.cache /app/.cache/huggingface && \
    chmod -R 777 /app/logs /app/backend/.cache /app/.cache

EXPOSE 7860

# Single worker with 4 threads shares the in-memory SBERT + O*NET tensors across requests
CMD ["sh", "-c", "gunicorn --chdir /app/backend -w 1 --threads 4 -t 180 -b 0.0.0.0:${PORT:-7860} api:app"]
