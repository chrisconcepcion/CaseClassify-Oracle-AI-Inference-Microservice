# Base image: NVIDIA CUDA 12.8 on Ubuntu 22.04 (Required for Blackwell architecture)
FROM nvidia/cuda:12.8.0-base-ubuntu22.04

# Python optimizations:
# 1. PYTHONUNBUFFERED=1: Flushes stdout immediately (better logging)
# 2. PYTHONDONTWRITEBYTECODE=1: Prevents .pyc file creation
# 3. PIP_NO_CACHE_DIR=1: Reduces image size by not caching downloads
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /code

# Install system dependencies
# Clean up apt lists immediately to keep image size small
RUN apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker layer caching
COPY requirements.txt .

# Install Python dependencies
# Using cu128 index to match requirements.txt for Blackwell optimization
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu128

# Copy application source code
COPY . .

# Expose port 8000 for the application
EXPOSE 8000

# Health check to ensure the service is running and GPU is visible
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Uvicorn server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]