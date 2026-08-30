FROM python:3.11-slim AS base

LABEL maintainer="officialarghya29"
LABEL description="CropSSL — Cross-Domain Robustness of SSL Vision Foundation Models for Crop Disease Detection"

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY . .

# Install package
RUN pip install --no-cache-dir -e .

EXPOSE 8000 8501

# Default: run API server
CMD ["python", "-m", "crop_ssl.backend.api"]
