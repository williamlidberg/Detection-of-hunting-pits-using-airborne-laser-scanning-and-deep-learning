FROM nvcr.io/nvidia/tensorflow:20.06-tf1-py3

RUN echo "Custom container downloaded!"
RUN apt-get -y update --fix-missing
RUN apt-get -y install libopencv-highgui-dev ffmpeg libsm6 libxext6 software-properties-common
COPY .  /bin
RUN echo "files copied to container"
RUN add-apt-repository -y ppa:ubuntugis/ppa
RUN apt-get update

RUN echo "Done installing packages in container!"

RUN pip install nvidia-pyindex
RUN pip install nvidia-tensorflow[horovod]
RUN pip install whitebox==2.0.3
RUN pip install pillow
RUN pip install opencv-python
RUN pip install tifffile
RUN pip install pandas
RUN pip install scikit-learn
RUN pip install geopandas
RUN pip install splitraster

RUN echo "Done installing conda packages in container!"
## old



