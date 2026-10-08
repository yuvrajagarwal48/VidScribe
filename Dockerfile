# =========================================================================
# VidScribe - Multimodal Video Intelligence & Agentic VideoRAG
# Production Dockerfile
# =========================================================================

FROM python:3.11-slim-bookworm

# Set environment flags
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    SERVER_HOST=0.0.0.0 \
    SERVER_PORT=8000

# Install critical system libraries:
# - ffmpeg: Required by MoviePy, Whisper & OpenCV for media decoding/encoding
# - tesseract-ocr: Required by pytesseract for on-screen text extraction
# - libgl1, libglib2.0-0: Required for headless OpenCV operations
# - curl: For container health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    tesseract-ocr \
    tesseract-ocr-eng \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python package dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-cache foundational ML models in the image layer to prevent cold-start download delays
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('base', device='cpu', compute_type='int8')" || \
    python -c "import whisper; whisper.load_model('base')"

# Copy application source code
COPY . .

# Ensure storage directories exist inside the container
RUN mkdir -p /app/data /app/processed/keyframes /app/processed/vectordb /app/summaries /app/reports

# Expose FastAPI & WebSocket service port (8000) and Streamlit dashboard port (8501)
EXPOSE 8000 8501

# Health check to verify service availability
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/videos || exit 1

# Launch VidScribe web application
CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8000"]
