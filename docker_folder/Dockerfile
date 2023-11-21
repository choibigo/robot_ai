FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04

ENV TZ=Asia/Seoul

RUN apt-get -y update
RUN apt-get install -y wget

RUN apt-get install -y gcc
RUN apt-get install -y g++
RUN apt-get install -y libgl1-mesa-glx
RUN apt-get install -y libglib2.0-0
RUN apt-get install -y tesseract-ocr tesseract-ocr-kor
RUN apt-get install -y x11-apps

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"
RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir /root/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh 
RUN conda init bash

SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
RUN pip install pybullet
RUN pip install numpy
RUN pip install gym
RUN pip install opencv-python
RUN pip install mediapipe
RUN pip install numpy==1.26.1
RUN pip install matplotlib==3.8.0
RUN pip install opencv-python
RUN pip install imageio==2.32.0
RUN pip install torch==2.1.1
RUN pip install torchvision==0.16.1
RUN pip install Pillow==10.0.1
RUN pip install scikit-build
RUN pip install scikit-image==0.21.0
RUN pip install scikit-learn>=0.23.2

RUN pip install ftfy regex tqdm
RUN apt-get -y update
RUN apt install git -y
RUN pip install git+https://github.com/openai/CLIP.git

# Display Option
ENV DISPLAY=$DISPLAY