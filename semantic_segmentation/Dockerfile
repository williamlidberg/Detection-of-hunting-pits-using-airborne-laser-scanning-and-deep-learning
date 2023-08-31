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
RUN mkdir /workspace/temp_inference
RUN mkdir /workspace/repo
COPY . /workspace/repo/
WORKDIR /workspace/code

