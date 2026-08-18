# ============================================================================
# PCB Defect Detection - Dockerfile
# One image, two selectable services:
#   - FastAPI  (default)       -> port 8000
#   - Streamlit (override CMD) -> port 8501
# ============================================================================

FROM python:3.11-slim AS base

# Prevent .pyc file generation and disable log buffering (real-time logs)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# ----------------------------------------------------------------------------
# System dependencies required by OpenCV / Ultralytics / video encoding
# libgl1 & libglib2.0-0 -> needed by cv2 (even the headless build sometimes needs them)
# ffmpeg -> for H.264 encoding of output videos (imageio-ffmpeg / cv2.VideoWriter)
# ----------------------------------------------------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        ffmpeg \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ----------------------------------------------------------------------------
# Install Python dependencies (kept separate from the app code copy to make
# better use of Docker layer caching)
# ----------------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ----------------------------------------------------------------------------
# Copy project code
# Note: model weights (*.onnx / *.pt) are intentionally NOT copied into the
# image; mount them at runtime instead (see the "Run with Docker" section in
# README.md).
# ----------------------------------------------------------------------------
COPY src/ ./src/
COPY app/ ./app/

# Run the container as a non-root user for better security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

# Both ports (FastAPI and Streamlit) are exposed; only the one started by CMD
# will actually be in use.
EXPOSE 8000 8501

ENV MODELS_DIR=/app/app/api/models \
    MODEL1_PATH=/app/app/api/models/best-pcb.onnx \
    MODEL2_PATH=/app/app/api/models/best_detect2.onnx \
    OUTPUT_DIR=/app/outputs \
    HOST=0.0.0.0 \
    PORT=8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:${PORT}/health || exit 1

# Default: run the FastAPI service.
# To run the Streamlit dashboard instead:
#   docker run ... <image> streamlit run app/streamlit_app/app.py --server.address=0.0.0.0 --server.port=8501
CMD ["uvicorn", "app.api.API_app:app", "--host", "0.0.0.0", "--port", "8000"]
