FROM mcr.microsoft.com/devcontainers/base:bookworm

# Install Python, zstd, and other prerequisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip zstd curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Pre-build the Python virtual environment with all dependencies
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/py_env && \
    /opt/py_env/bin/pip install --no-cache-dir -r /tmp/requirements.txt && \
    /opt/py_env/bin/pip uninstall -y \
        nvidia-cublas-cu12 nvidia-cuda-cupti-cu12 \
        nvidia-cuda-nvrtc-cu12 nvidia-cuda-runtime-cu12 \
        nvidia-cufft-cu12 nvidia-cufile-cu12 \
        nvidia-curand-cu12 nvidia-cusparse-cu12 \
        nvidia-cusparselt-cu12 nvidia-nccl-cu12 \
        nvidia-nvjitlink-cu12 nvidia-nvshmem-cu12 \
        nvidia-nvtx-cu12 2>/dev/null || true && \
    /opt/py_env/bin/pip install --no-cache-dir \
        torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cpu

# Pre-download the sentence-transformers embedding model
RUN /opt/py_env/bin/python -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Pre-install Ollama (binary only — no model pull)
RUN curl -fsSL https://ollama.com/install.sh | sh
