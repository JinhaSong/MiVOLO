FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu20.04

ENV TZ Asia/Seoul
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get -y install \
    python3 python3-pip python3-dev \
    libgl1 tzdata git wget ssh vim

RUN ln -sf /usr/share/zoneinfo/Asia/Seoul /etc/localtime
RUN ln -s /usr/bin/python3 /usr/bin/python
RUN echo "root:password" | chpasswd
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes #prohibit-password/' /etc/ssh/sshd_config

WORKDIR /workspace
ADD . .

RUN pip3 install --upgrade pip
RUN pip3 install setuptools
RUN pip3 install torch==1.13.1+cu117 torchvision==0.14.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
RUN pip3 install -r requirements.txt
