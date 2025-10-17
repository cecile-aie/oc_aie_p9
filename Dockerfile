# ============================================================
# 🧠 Base image : PyTorch 2.4.0 + CUDA 12.4 + cuDNN 9
# ============================================================
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

# ============================================================
# 🔧 1. System dependencies
# ============================================================
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    nano \
    build-essential \
    python3-opencv \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# ============================================================
# 📦 2. Python dependencies
# ============================================================
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

RUN pip install --no-cache-dir \
    numpy==2.2.6 \
    pandas \
    matplotlib \
    seaborn \
    scikit-learn==1.6.1 \
    scikit-image \
    tqdm \
    jupyterlab \
    fastparquet \
    plotly \
    SimpleITK \
    umap-learn \
    scikit-posthocs \
    statsmodels \
    nvidia-ml-py3

# Torch utils + vision
RUN pip install --no-cache-dir \
    torchvision==0.19.0 \
    torchaudio==2.4.0

# ============================================================
# 🎨 3. Histopathology preprocessing (Torch-StainTools)
# ============================================================
# ⚡ GPU compatible stain normalization (alternative à torchstain)
RUN pip install --no-cache-dir \
    torch-staintools==1.0.4 \
    opencv-python-headless==4.10.0.84

# (widgets + pillow)
RUN pip install --no-cache-dir \
    ipywidgets \
    pillow

# ============================================================
# 🧭 4. Workspace
# ============================================================
WORKDIR /workspace
ENV PYTHONPATH=/workspace


# ============================================================
# 🎯 Default command
# ============================================================
# CMD ["bash"]


# ---------------------------------------------------------------
# Configuration JupyterLab
# ---------------------------------------------------------------
EXPOSE 8888
CMD ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
