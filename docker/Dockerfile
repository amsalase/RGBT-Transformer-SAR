FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    JUPYTER_RUNTIME_DIR=/tmp/jupyter-runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    curl \
    vim \
    nano \
    unzip \
    build-essential \
    cmake \
    pkg-config \
    ca-certificates \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/rgbt-requirements.txt

RUN python -m pip install --no-cache-dir \
        -r /tmp/rgbt-requirements.txt \
    && rm -f /tmp/rgbt-requirements.txt

RUN python -m ipykernel install \
    --sys-prefix \
    --name rgbt \
    --display-name "Python (RGB-T Transformer)"

WORKDIR /workspace

EXPOSE 8898

CMD ["/bin/bash"]
