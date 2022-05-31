FROM nvcr.io/nvidia/tensorflow:20.12-tf2-py3

RUN echo "Custom container downloaded!"
RUN apt-get -y update --fix-missing
RUN apt-get -y install libopencv-highgui-dev ffmpeg libsm6 libxext6 software-properties-common
RUN pip install --upgrade pip

RUN pip install whitebox==2.0.3
#RUN pip install pillow
RUN pip install matplotlib==3.5.1 
RUN pip install opencv-python==4.5.5.64 
RUN pip install tifffile==2022.4.26
RUN pip install pandas==1.4.2 
RUN pip install scikit-learn==1.0.2
RUN pip install geopandas==0.10.2 
RUN pip install splitraster==0.3.2
#RUN pip install imageio==2.15.0
RUN pip install rasterio==1.2.10 
RUN pip install leafmap==0.9.1
RUn pip install rtree==1.0.0
RUN pip install torch
RUN pip install torchvision
RUN pip install jupyter_contrib_nbextensions
RUN pip install jupyter_nbextensions_configurator
RUN jupyter nbextensions_configurator enable --user
RUN echo "Installed python packages!"

RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
RUN apt-get install gdal-bin -y
RUN apt-get install libgdal-dev -y
RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
RUN export C_INCLUDE_PATH=/usr/include/gdal
RUN pip install GDAL
RUN echo "Gdal installed!"

RUN mkdir -p ~/workspace/temp_dir
