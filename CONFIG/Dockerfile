FROM python:3.10-slim

LABEL maintainer="research@project"
LABEL description="Bias Evaluation of CAD in Hate Speech Detection — ACM MM 2026"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PYTHONPATH=/app
ENV TRANSFORMERS_CACHE=/app/.cache/transformers
ENV SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers

# Default: run full smoke test to verify environment
CMD ["python", "-m", "pytest", "tests/", "-v", "--tb=short", "-x"]
