FROM tensorflow/tensorflow:2.7.0-gpu
#FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04
RUN echo "Custom container downloaded!"
RUN apt-get -y update --fix-missing
RUN apt-get -y install libopencv-highgui-dev ffmpeg libsm6 libxext6 software-properties-common
COPY .  /bin
RUN echo "files copied to container"
