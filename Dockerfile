FROM nvidia/cuda:9.0-cudnn7-devel-ubuntu16.04
MAINTAINER hirochika-f

WORKDIR /root

# Apt update
RUN apt-get update && apt-get install -y sudo
#RUN sudo sed -i -e 's/archive.ubuntu.com\|security.ubuntu.com/old-releases.ubuntu.com/g' /etc/apt/sources.list
RUN sudo apt-get update && sudo apt-get upgrade -y

# Install Python and Image processing
RUN sudo apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libgtk2.0-dev \
    libjpeg-dev \
    libpng-dev \
    python3-dev \
    python3-pip \
    python3-setuptools

# Install PyTorch
RUN pip3 install pip==18.0
RUN pip3 install numpy
RUN pip3 install torch==0.4.0 && pip3 install torchvision==0.2.1

# Download pix2pix
RUN git clone https://github.com/GINK03/pytorch-pix2pix

# Copy original dataset

