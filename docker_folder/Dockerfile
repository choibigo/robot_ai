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

# Display Option
ENV DISPLAY=$DISPLAY