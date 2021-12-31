FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3
#FROM tensorflow/tensorflow:2.7.0-gpu
#FROM mcr.microsoft.com/azureml/openmpi3.1.2-cuda10.1-cudnn7-ubuntu18.04
RUN echo "Custom container downloaded!"
RUN apt-get -y update --fix-missing
RUN apt-get -y install libopencv-highgui-dev ffmpeg libsm6 libxext6 software-properties-common
COPY .  /bin

RUN pip install --upgrade pip
RUN pip install tifffile
RUN pip install imagecodecs
RUN pip install whitebox
RUN pip install opencv-python
RUN pip install opencv-python
RUN pip install pandas
RUN pip install sklearn
RUN pip install geopandas
RUN pip install splitraster

RUN echo "files copied to container"
# Install dependencis for gdal
#RUN add-apt-repository -y ppa:ubuntugis/ppa
#RUN apt-get update
#RUN apt-get -y install gdal-bin libgdal-dev

#RUN echo "Done installing packages in container!"

#RUN conda update -n base -c defaults conda
#RUN conda install -c anaconda opencv
#RUN conda install -c anaconda pillow
#RUN conda install -c conda-forge tifffile
#RUN conda install -c anaconda pandas
#RUN conda install -c conda-forge scikit-learn
#RUN conda install -c conda-forge imagecodecs

#ENV CPLUS_INCLUDE_PATH=/usr/include/gdal
#ENV C_INCLUDE_PATH=/usr/include/gdal

# Install gdal with pip
#RUN pip install GDAL==2.4.2

RUN echo "Done installing conda packages in container!"
