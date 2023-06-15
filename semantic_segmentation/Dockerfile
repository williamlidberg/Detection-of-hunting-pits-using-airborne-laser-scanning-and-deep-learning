FROM nvcr.io/nvidia/tensorflow:22.04-tf2-py3

RUN apt-get update
# install dependencies for opencv
RUN apt-get install -y ffmpeg libsm6 libxext6

# setup GDAL
RUN apt-get install -y software-properties-common
RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
RUN apt-get install -y gdal-bin
RUN apt-get install -y libgdal-dev
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal
RUN pip install GDAL

RUN pip install opencv-python
RUN pip install matplotlib
RUN pip install tifffile
RUN pip install geopandas
RUN pip install imagecodecs
RUN pip install whitebox
# added to install splitraster witout numpy version conclict
RUN pip install imageio==2.15.0
RUN pip install splitraster
RUN pip install rvt-py
RUN pip install torch torchvision
RUN pip install seaborn
RUN pip install scikit-learn
RUN pip install rasterio
RUN pip install jupyterlab

# create mount points for data and source code in container's start directory
RUN mkdir /workspace/data
RUN mkdir /workspace/code
RUN mkdir /workspace/temp

WORKDIR /workspace/code



# FROM tensorflow/tensorflow:latest-gpu-jupyter

# RUN echo "Custom container downloaded!"
# RUN apt-get -y update --fix-missing
# RUN apt-get -y install libopencv-highgui-dev ffmpeg libsm6 libxext6 software-properties-common
# RUN pip install --upgrade pip

# RUN pip install whitebox
# RUN pip install matplotlib==3.5.1 
# RUN pip install opencv-python==4.5.5.64 
# RUN pip install tifffile==2022.4.26
# RUN pip install pandas==1.4.2 
# RUN pip install scikit-learn==1.0.2
# RUN pip install geopandas==0.10.2 
# RUN pip install splitraster==0.3.2
# RUN pip install rasterio==1.2.10 
# RUN pip install leafmap==0.9.1
# RUN pip install rtree==1.0.0
# RUN pip install torch
# RUN pip install torchvision
# RUN pip install jupyterlab
# RUN pip install seaborn
# RUN echo "Installed python packages!"

# RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
# RUN apt-get install gdal-bin -y
# RUN apt-get install libgdal-dev -y
# RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
# RUN export C_INCLUDE_PATH=/usr/include/gdal
# RUN pip install GDAL
# RUN echo "Gdal installed!"

# RUN mkdir -p /workspace/temp
