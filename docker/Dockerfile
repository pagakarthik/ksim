FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies for Python build and add deadsnakes PPA
RUN apt-get update && \
    apt-get install -y software-properties-common build-essential && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update

# Install Python 3.11 and related packages
RUN apt-get install -y python3.11 python3.11-venv python3.11-distutils python3.11-dev python3-pip

# Set python3 to point to python3.11
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 2 && \
    update-alternatives --set python3 /usr/bin/python3.11

# Upgrade pip for Python 3.11
RUN python3 -m pip install --upgrade pip setuptools

# Reset frontend to default (optional, for better logs in later steps)
ENV DEBIAN_FRONTEND=dialog

# (Optional) Check version
RUN python3 --version

# RUN apt-get update && apt-get install -y python3 python3-pip

# RUN pip3 install --upgrade pip

# Install JAX with CUDA 11 support
RUN pip install distrax==0.1.3
RUN pip install chex==0.1.6 optax==0.1.3 equinox==0.9.0

RUN pip install --upgrade jax==0.4.13
RUN pip install --upgrade https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.4.13+cuda11.cudnn86-cp311-cp311-manylinux2014_x86_64.whl


RUN apt-get update && apt-get install -y git

RUN apt-get update && \
    apt-get install -y \
        libgl1 \
        libopengl0 \
        libglib2.0-0 \
        libx11-6 \
        libxext6 \
        libsm6 \
        libxrender1 \
        libxi6 \
        libxrandr2 \
        libxcursor1 \
        libxinerama1 \
        libxxf86vm1 \
        libxfixes3 \
        libglfw3 \
        && rm -rf /var/lib/apt/lists/*

# use local version
# WORKDIR /ksim
# COPY ../ /ksim
# RUN pip install -e .

# use pre-built lib from pip repo
RUN pip install ksim
