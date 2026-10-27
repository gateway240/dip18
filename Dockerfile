FROM songhesd/cuda:8.0-cudnn6-devel-ubuntu16.04

ENV DEBIAN_FRONTEND=noninteractive

# System packages
RUN apt-get update && apt-get install -y \
    software-properties-common \
    build-essential \
    wget \
    curl \
    git \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    pkg-config \
    python3.5 \
    python3.5-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

# Make Python 3.5 the default
RUN ln -sf /usr/bin/python3.5 /usr/bin/python

# Upgrade pip to the newest version that still supports Python 3.5
RUN python -m pip install --upgrade "pip<21"

# TensorFlow 1.4 GPU
RUN pip install \
    https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl

RUN pip install \
    opencv-python==3.4.5.20 \
    llvmlite==0.31.0 \
    numba==0.47.0 \
    numpy-quaternion==2019.12.11.22.25.52 \
    numpy~=1.15 \
    scipy

CMD ["/bin/bash"]