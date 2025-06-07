FROM python:3.10-slim AS base

# Optimized environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PYTHONFAULTHANDLER=1

# Install runtime system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libpq5 \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y

# === DEPENDENCY INSTALLATION STAGE ===
FROM base AS dependencies

# Install build tools (only needed during build)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /tmp

# Copy requirements.txt FIRST (for optimal Docker layer caching)
COPY requirements.txt .

# Create virtual environment for clean package management
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt && \
    # Add google-cloud-storage for model download (replacing gcloud CLI)
    pip install --no-cache-dir google-cloud-storage

# === PRODUCTION STAGE ===
FROM base AS production

# Create non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Set working directory
WORKDIR /app
RUN chown appuser:appuser /app

# Copy only the virtual environment (much smaller than all site-packages)
COPY --from=dependencies /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Create application directories
RUN mkdir -p credentials ckpt image/routes image/services image/reverse_image_search && \
    chown -R appuser:appuser /app

# Create model download script using Python instead of gsutil
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
# Authenticate with Google Cloud\n\
if [ -f "$GOOGLE_APPLICATION_CREDENTIALS" ]; then\n\
    # Download models using Python SDK instead of gsutil\n\
    echo "📥 Downloading models from Google Cloud Storage..."\n\
    \n\
    python3 -c "\n\
import os\n\
from google.cloud import storage\n\
\n\
try:\n\
    client = storage.Client()\n\
    bucket = client.bucket(\"focust-dev-cloud-storage\")\n\
    blob = bucket.blob(\"checkpoints/face-deepfake-detection-epoch=07-val_metric=1.000.ckpt\")\n\
    blob.download_to_filename(\"/app/ckpt/face-deepfake-detection-epoch=07-val_metric=1.000.ckpt\")\n\
    print(\"✅ Model download complete\")\n\
except Exception as e:\n\
    print(f\"❌ Error downloading model: {e}\")\n\
    exit(1)\n\
"\n\
else\n\
    echo "⚠️ Service account key not found"\n\
    exit 1\n\
fi\n\
' > /app/download_models.sh && \
chmod +x /app/download_models.sh && \
chown appuser:appuser /app/download_models.sh

# Create entrypoint script that downloads models then starts the app
RUN echo '#!/bin/bash\n\
set -e\n\
\n\
echo "🚀 Starting application..."\n\
\n\
# Check if models exist, if not download them\n\
if [ ! -f "/app/ckpt/face-deepfake-detection-epoch=07-val_metric=1.000.ckpt" ]; then\n\
    echo "📥 Models not found, downloading..."\n\
    /app/download_models.sh\n\
else\n\
    echo "✅ Models already exist, skipping download"\n\
fi\n\
\n\
# Start the main application\n\
echo "🌟 Starting main application..."\n\
exec python main.py\n\
' > /app/entrypoint.sh && \
chmod +x /app/entrypoint.sh && \
chown appuser:appuser /app/entrypoint.sh

# Switch to non-root user
USER appuser

# Copy application code (do this LAST as it changes most frequently)
COPY --chown=appuser:appuser . .

# Expose port
EXPOSE 5500

# Optimized health check
HEALTHCHECK --interval=60s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5500/health || exit 1

# Use entrypoint script instead of directly running main.py
CMD ["/app/entrypoint.sh"]
