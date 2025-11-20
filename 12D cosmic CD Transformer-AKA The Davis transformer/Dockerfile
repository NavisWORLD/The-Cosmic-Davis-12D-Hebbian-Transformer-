# Multi-stage Dockerfile for 12D Cosmic Synapse Transformer
# Optimized for production deployment

FROM python:3.10-slim as base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data checkpoints logs

# Expose port for API server
EXPOSE 5000

# Default command: run API server
CMD ["python", "inference_cosmic_transformer.py", "serve", "checkpoints/model.pt", "--host", "0.0.0.0", "--port", "5000"]

# Development stage
FROM base as development

# Install development dependencies
COPY requirements-dev.txt .
RUN pip install --no-cache-dir -r requirements-dev.txt

# Jupyter notebook port
EXPOSE 8888

CMD ["/bin/bash"]

# Production stage
FROM base as production

# Run as non-root user for security
RUN useradd -m -u 1000 cosmic && \
    chown -R cosmic:cosmic /app

USER cosmic

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/health')" || exit 1
