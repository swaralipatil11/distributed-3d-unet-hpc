# Stage 1: Build dependencies in a Python environment
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system utilities needed to build certain wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the requirements to cache dependency layer
COPY requirements.txt .

# Install dependencies to user local directory to preserve caching
RUN pip install --no-cache-dir --user -r requirements.txt


# Stage 2: Final lightweight execution environment with official CUDA support
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04 AS runner

WORKDIR /app

# Install Python runtime environment
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3-pip \
    python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Link python and python3 to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Copy dependencies from builder stage
COPY --from=builder /root/.local /root/.local

# Set environment path to expose installed wheels (uvicorn, torch, monai, etc)
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app:$PATH

# Copy src and deploy assets
COPY src/ /app/src/
COPY deploy/ /app/deploy/

# Create a non-root application user for improved runtime security
RUN useradd -u 8888 appuser && chown -R appuser:appuser /app
USER appuser

# Expose FastAPI default port
EXPOSE 8000

# Environment variables
ENV KAFKA_BOOTSTRAP_SERVERS=localhost:9092
ENV KAFKA_TOPIC=mri-inference-requests

# Start the uvicorn worker service
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
