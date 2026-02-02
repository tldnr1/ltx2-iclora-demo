# =============================================================================
# LTX-2 IC-LoRA Demo - Dockerfile
# =============================================================================

FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# CUDA memory optimization
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Gradio settings
ENV GRADIO_SERVER_NAME=0.0.0.0
ENV GRADIO_SERVER_PORT=7860

# =============================================================================
# System Dependencies
# =============================================================================

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# =============================================================================
# Python Dependencies
# =============================================================================

WORKDIR /app

# Copy requirements first for caching
COPY requirements.txt .

# Install base requirements
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Install LTX-2 packages from GitHub
# Using specific commit/tag for reproducibility
RUN pip install --no-cache-dir --no-build-isolation \
    'git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-core' && \
    pip install --no-cache-dir --no-build-isolation \
    'git+https://github.com/Lightricks/LTX-2.git#subdirectory=packages/ltx-pipelines'

# Optional: Install xformers for attention optimization
# RUN pip install --no-cache-dir xformers

# =============================================================================
# Application
# =============================================================================

# Copy application code
COPY config.py .
COPY preprocessing.py .
COPY pipeline.py .
COPY app.py .

# Create directories
RUN mkdir -p /app/outputs /app/uploads

# Models will be mounted at runtime
VOLUME ["/models"]

# =============================================================================
# Entrypoint
# =============================================================================

EXPOSE 7860

CMD ["python", "app.py"]
