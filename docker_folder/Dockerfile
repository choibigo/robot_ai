FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
 
# Remove any third-party apt sources to avoid issues with expiring keys.
RUN rm -f /etc/apt/sources.list.d/*.list
 
# Install some basic utilities & python prerequisites
RUN apt-get update -y && apt-get install -y --no-install-recommends\
    wget vim curl ssh tree sudo git zip unzip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
 
# Set up time zone
ENV TZ=Asia/Seoul
RUN sudo ln -snf /usr/share/zoneinfo/$TZ /etc/localtime

RUN apt-get update


# libGL error: No matching fbConfigs or visuals found 관련
# sudo apt-get install software-properties-common
# sudo add-apt-repository ppa:ubuntu-x-swat/updates -y 
# sudo add-apt-repository ppa:ubuntu-x-swat/updates -y

# export LIBGL_ALWAYS_INDIRECT=1
# sudo apt-get install -y mesa-utils libgl1-mesa-glx


# Failed to create GL 3.3 context ... using old-style GLX context 관련
# conda install -c conda-forge libgcc=5.2.0 -y
# conda install -c anaconda libstdcxx-ng -y
# conda install -c conda-forge gcc=12.1.0