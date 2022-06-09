FROM ultralytics/yolov5:latest

RUN pip install --upgrade pip

RUN pip install tifffile==2022.4.26
RUN pip install pybboxes
RUN pip install whitebox==2.0.3
RUN pip install geopandas==0.10.2 
RUN pip install splitraster==0.3.2
RUN pip install rasterio==1.2.10 
RUN pip install leafmap==0.9.1

RUN mkdir -p ~/workspace/temp