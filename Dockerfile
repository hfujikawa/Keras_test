# Dockerfile to setup a build environment for TensorFlow
# using Intel MKL and Anaconda3 Python
# GPU support with CUDA 9.1 and cudnn7.1

FROM nvidia/cuda:9.1-cudnn7-devel-ubuntu16.04

MAINTAINER nobody, not even me

# Add a few needed packages to the base Ubuntu 16.04
# OK, maybe *you* don't need emacs :-)
RUN \
    apt-get update && apt-get install -y \
    build-essential \
    curl \
    emacs-nox \
    git \
    openjdk-8-jdk \
    zlib1g-dev \
    bash-completion \
    && rm -rf /var/lib/lists/*

# Use version 11.1 bazel! install from the deb file.
COPY bazel_0.11.1-linux-x86_64.deb /root/
RUN \
  cd /root; dpkg -i bazel_0.11.1-linux-x86_64.deb && \
  rm -f bazel_0.11.1-linux-x86_64.deb

# Copy in and install Anaconda3 from the shell archive
# Anaconda3-5.1.0-Linux-x86_64.sh
COPY Anaconda3* /root/
RUN \
  cd /root; chmod 755 Anaconda3*.sh && \
  ./Anaconda3*.sh -b && \
  echo 'export PATH="$HOME/anaconda3/bin:$PATH"' >> .bashrc && \
  rm -f Anaconda3*.sh

# Copy in the CUDA configuration files
COPY cuda.sh /etc/profile.d/
COPY cuda.conf /etc/ld.so.conf.d/

# That's it! That should be enough to do a TensorFlow 1.7 GPU build
# using CUDA 9.1 Anaconda Python 3.6 Intel MKL with gcc 5.4
