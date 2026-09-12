# ── PULMO·AI — Hugging Face Spaces Docker Image ──────────────────────────────
# Base: slim Python 3.11 (matches project requirement; avoids heavy CUDA bloat)
FROM python:3.11-slim

# HF Spaces runs containers as a non-root user (uid 1000).
# Create the user early so all file ownership is correct.
RUN useradd -m -u 1000 appuser

WORKDIR /app

# ── System deps ───────────────────────────────────────────────────────────────
# libgl1 + libglib2.0-0 are required by opencv-python headless
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (install before copying code for better layer caching) ────────
COPY requirements.txt .

# Use opencv-python-headless (no GUI) — smaller and works in containers
RUN pip install --no-cache-dir --upgrade pip \
    && sed 's/opencv-python>=/opencv-python-headless>=/g' requirements.txt \
       > requirements_docker.txt \
    && pip install --no-cache-dir -r requirements_docker.txt \
    && rm requirements_docker.txt

# ── Copy application source ───────────────────────────────────────────────────
COPY --chown=appuser:appuser src/        ./src/
COPY --chown=appuser:appuser models/     ./models/
COPY --chown=appuser:appuser data/test/  ./data/test/
COPY --chown=appuser:appuser run_app.py  .
COPY --chown=appuser:appuser main.py     .
COPY --chown=appuser:appuser setup.py    .

# ── Switch to non-root ────────────────────────────────────────────────────────
USER appuser

# ── Runtime config ────────────────────────────────────────────────────────────
# HF Spaces injects $PORT (usually 7860). run_app.py reads it.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860

EXPOSE 7860

CMD ["python", "run_app.py"]
