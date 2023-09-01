FROM anibali/pytorch:1.8.1-cuda11.1-ubuntu20.04

USER root
#RUN echo "deb [by-hash=no] http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/cuda.list && rm /etc/apt/sources.list.d/nvidia-ml.list
#RUN sudo apt-get update
ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /mnt/

COPY ./ ./

#ENV REQUESTS_CA_BUNDLE cacert.pem
RUN apt-get -y update --fix-missing
RUN apt-get install ffmpeg libsm6 libxext6  -y
RUN apt-get -y install gdal-bin
RUN apt-get -y install libgdal-dev
RUN apt-get -y install g++
RUN pip install gdal==$(ogrinfo --version | awk -F " " '{print $2}' | awk -F "," '{print $1}')

RUN pip install -r requirements.txt


RUN echo "Done installing packages in container!"
