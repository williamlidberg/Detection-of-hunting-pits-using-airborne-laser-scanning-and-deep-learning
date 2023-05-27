FROM ultralytics/yolov5:latest
RUN pip install --upgrade pip

RUN pip install tifffile==2022.4.26
RUN pip install	pybboxes==0.0.1
RUN pip install	whitebox
RUN pip install	geopandas==0.10.2
RUN pip install	splitraster==0.3.2
RUN pip install	rasterio
RUN pip install tensorflow
#RUN DEBIAN_FRONTEND=noninteractive apt-get install software-properties-common -y
#RUN add-apt-repository ppa:ubuntugis/ppa && apt-get update
#RUN apt-get install gdal-bin -y
#RUN apt-get install libgdal-dev -y
#RUN export CPLUS_INCLUDE_PATH=/usr/include/gdal
#RUN export C_INCLUDE_PATH=/usr/include/gdal
##RUN echo "Gdal installed!"

RUN mkdir -p /workspace/temp