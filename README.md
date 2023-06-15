[![docs](https://img.shields.io/badge/whitebox-docs-brightgreen.svg)](https://www.whiteboxgeo.com/manual/wbt_book/preface.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter Follow](https://img.shields.io/twitter/follow/William_Lidberg?style=social)](https://twitter.com/william_lidberg)



# Cultural-remains
## AIM
The aim of this project is to evaluate methods to automatically detect trapping pits using remote sensing. Two machine learning methods will be tested on multiple topographical indices extracted from LiDAR data. 


<img src="images/träd9.png" alt="Charcoal kiln" width="50%"/>


# Table of  Contents

1. [Docker containers](#Docker-containers)
3. [Training data](#Training-data)
    1. [Create digital elevation model](##Create-digital-elevation-model)
    2. [Extract and normalize topographical indices](##Extract-and-normalize-topographical-indices)
6. [Semantic segmentation](#Semantic-segmentation)
    1. [Create segmentation masks](##Create-segmentation-mask)
    2. [Create image chips](##Create-image-chips)
    3. [Train U-net](##Train-U-net)
    4. [Evaluate U-net](##Evaluate-U-net)
    5. [Inference U-net](##Inference-U-net)
    6. [Post-processing U-net](##Post-processing-U-net)
7. [Object detection](#Object-detection)
    1. [Create bounding boxes](##Create-bounding-boxes)
    2. [Train YOLO](##Train-YOLO)
    3. [Evaluate YOLO](##Evaluate-YOLO)
    4. [Inference YOLO](##Inference-YOLO)
8. [Transfer learning](#Transfer-learning)
    1. [Data description](##Data-description)
    2. [Select craters](##Select-craters)
    3. [Craters to segmentation masks](##Craters-to-segmentation-masks)
    4. [Craters to bounding boxes](##Craters-to-bounding-boxes)
9. [References](#References)

***


# Docker containers
Docker containers will be used to manage all envrionments in this project. Different images were used for segmentation and object detection\
**Segmentation:**
semantic_segmentation\Dockerfile
\
**Object detection:**
object_detection\Dockerfile

Navigate to respective dockerfile in the segmentation or object detection directories and build the containers

    docker build -t segmentation .
    docker build -t detection .


You can run the container in the background with screen

    screen -S segmentation



**Without notebook**

mig 1
        docker run -it --gpus device=0:0 -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/Extension_100TB/national_datasets/laserdataskog/:/workspace/lidar segmentation:latest bash

        

mig 2
        docker run -it --gpus device=0:1 -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/Extension_100TB/national_datasets/laserdataskog/:/workspace/lidar segmentation:latest bash



    docker run -it --rm -p 8882:8882 --gpus all -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/Extension_100TB/national_datasets/laserdataskog/:/workspace/lidar segmentation:latest bash

    docker run -it --gpus all -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/Extension_100TB/national_datasets/laserdataskog/:/workspace/lidar segmentation:latest bash

    cd /workspace/code/notebooks/

    jupyter lab --ip=0.0.0.0 --port=8887 --allow-root --no-browser --NotebookApp.allow_origin='*'

    ssh -L 8887:localhost:8887 william@193.10.101.143


    ssh -L 8881:localhost:8881 wmli0001@lidar1-1.ad.slu.se


# Training data
The training data were collected from multiple sources. Historical forest maps from local archives where digitized and georeferenced. Open data from the [swedish national heritage board were downloaded and digitized](https://pub.raa.se/). All remains where referenced with the liDAR data in order to match the reported remain to the LiDAR data. In total x hunting pits where manually digitized and corrected this way.

## Create digital elevation model
The laser data contained 1-2 points / m2 and can be downloaded from Lantmäteriet: https://www.lantmateriet.se/en/geodata/geodata-products/product-list/laser-data-download-forest/. The Laser data is stored as .laz tiles where each tile is 2 500 m x 2 500 m

**Select lidar tiles based on location of training data**\
First pool all laz files in a single directory

    python /workspace/code/tools/pool_laz.py

Then Create a shapefile tile index of all laz tiles in the pooled directory

    python /workspace/code/tools/lidar_tile_footprint.py /workspace/lidar/pooled_laz_files/ /workspace/data/footprint.shp

Use the shapefile tile index and a shapefile of all field data to create a polygon that can be used to select and copy relevant laz tilesto a new directory

    python /workspace/code/tools/copy_laz_tiles.py /workspace/data/footprint.shp /workspace/code/data/Hunting_pits_covered_by_lidar.shp /workspace/lidar/pooled_laz_files/ /workspace/data/selected_lidar_tiles_pits/



Finally use whitebox tools to create digital elevation models from the selected lidar data
**Training and testing areas**
    python /workspace/code/tools/laz_to_dem.py /workspace/data/selected_lidar_tiles_pits/ /workspace/data/dem_tiles_pits/ 0.5

    python /workspace/code/tools/laz_to_dem.py /workspace/data/selected_lidar_tiles_pits/ /workspace/data/dem_tiles_pits_1m/ 1.0

    python /workspace/code/tools/laz_to_dem.py /workspace/data/selected_lidar_tiles_pits/ /workspace/data/dem_tiles_pits_test/ 1.0

<br/>

## Extract and normalize topographical indices
Training a model directly on the digital elevation model is not practical since the values ranges from 0 to 2000 m. Instead different topographical indices were extracted from the DEM. The topographical data will be the same for both segmentation and object detection. All topographical indices are extracted using [Whitebox Tools](https://www.whiteboxgeo.com/manual/wbt_book/preface.html). The indices used are:  
*   [Multidirectionl hillshade](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html?highlight=multidirec#multidirectionalhillshade)
*   [Max elevation Deviation](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html?highlight=maxelevation#maxelevationdeviation)
*   [Multiscale Elevation Percentile](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html?highlight=multiscale#multiscaleelevationpercentile)
*   [Min curvature](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html?highlight=min%20curvature#minimalcurvature)
*   [Max curvature](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html?highlight=max%20curvature#maximalcurvature)
*   [Profile curvature](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html?highlight=profile#profilecurvature)
*   [Spherical Standard deviation of normals](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html?highlight=min%20curvature#SphericalStdDevOfNormals)
*   [Multiscale standard deviation of normals](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html?highlight=multiscale#multiscalestddevnormals)
*   [Elevation above pit](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/geomorphometric_analysis.html?highlight=elevation%20above%20pit#elevabovepit)
*   [Depth In Sink](https://www.whiteboxgeo.com/manual/wbt_book/available_tools/hydrological_analysis.html?highlight=depth#depthinsink)


This script extracts the topographical indices and normalizes them between 0 and 1. This step takes around 30 seconds / tile. It would take about 26 days for Sweden on 0.5 m resolution.

    python /workspace/code/Extract_topographcical_indices_05m.py /workspace/temp/ /workspace/data/dem_tiles_pits/ /workspace/data/topographical_indices_normalized_pits/hillshade/ /workspace/data/topographical_indices_normalized_pits/maxelevationdeviation/ /workspace/data/topographical_indices_normalized_pits/multiscaleelevationpercentile/ /workspace/data/topographical_indices_normalized_pits/minimal_curvature/ /workspace/data/topographical_indices_normalized_pits/maximal_curvature/ /workspace/data/topographical_indices_normalized_pits/profile_curvature/ /workspace/data/topographical_indices_normalized_pits/stdon/ /workspace/data/topographical_indices_normalized_pits/multiscale_stdon/ /workspace/data/topographical_indices_normalized_pits/elevation_above_pit/ /workspace/data/topographical_indices_normalized_pits/depthinsink/

    python /workspace/code/Extract_topographcical_indices_1m.py /workspace/temp/ /workspace/data/dem_tiles_pits_1m/ /workspace/data/topographical_indices_normalized_pits_1m/hillshade/ /workspace/data/topographical_indices_normalized_pits_1m/maxelevationdeviation/ /workspace/data/topographical_indices_normalized_pits_1m/multiscaleelevationpercentile/ /workspace/data/topographical_indices_normalized_pits_1m/minimal_curvature/ /workspace/data/topographical_indices_normalized_pits_1m/maximal_curvature/ /workspace/data/topographical_indices_normalized_pits_1m/profile_curvature/ /workspace/data/topographical_indices_normalized_pits_1m/stdon/ /workspace/data/topographical_indices_normalized_pits_1m/multiscale_stdon/ /workspace/data/topographical_indices_normalized_pits_1m/elevation_above_pit/ /workspace/data/topographical_indices_normalized_pits_1m/depthinsink/
    
<img src="images/distribution.PNG" alt="Distribution of normalized topographical indicies" width="50%"/>\
Distribution of the normalised topographical indices from one tile.

# Semantic segmentation
Semantic segmentation uses masks where each pixel in the mask coresponds to a class. In our case the classes are:

0. Background values
1. Hunting pits

    

<img src="images/Hunting_kids.jpg" alt="Study area" width="75%"/>\
The left image is a hunting pit (kids for scale) and the right image is the same hunting pit in the digital elevation model.


## Create segmentation masks
The training data is stored as digitized polygons where each feature class is stored in the column named "classvalue". Note that only polygons overlapping a dem tile will be converted to a labeled tile. polygons outside of dem tiles are ignored.

    python /workspace/code/tools/create_segmentation_masks.py /workspace/data/dem_tiles_pits/ /workspace/code/data/Hunting_pit_polygons.shp Classvalue /workspace/data/segmentation_masks_pits_05m/

    python /workspace/code/tools/create_segmentation_masks.py /workspace/data/dem_tiles_pits_1m/ /workspace/code/data/Hunting_pit_polygons.shp Classvalue /workspace/data/segmentation_masks_pits_1m/
    
## Create image chips
Each of the 2.5km x 2.5km dem tiles were Split into smaller image chips with the size 256 x 256 pixels. This corresponds to 125m x 125m in with a 0.5m DEM resolution.
```diff
- Make sure the directory is empty/new so the split starts at 1 each time
```
python /workspace/code/tools/split_training_data.py /workspace/data/segmentation_masks_pits_1m/ /workspace/data/split_data_pits_1m/labels/ --tile_size 250

The bash script ./code/split_indices.sh will remove and create new directories and then run the splitting script on all indicies. Each 2.5 km x 2.5 km tile is split into image chips with the size 250 x 250 pixels.

    ./workspace/code/split_indices_05m.sh

    ./workspace/code/split_indices_1m.sh



## Create bounding boxes
Bounding boxes can be created from segmentation masks if each object has a uniqe ID. A different column in the shapefile was used to achive this : object_id. Run the "create_segmentation_mask.py script on this column.

**Create new segmentation masks with uniqe ID**


    python /workspace/code/tools/create_segmentation_masks.py /workspace/data/dem_tiles_pits/ /workspace/code/data/Hunting_pits_covered_by_lidar.shp object_id /workspace/data/object_detection/segmentation_masks_tiles_05m/
    python /workspace/code/tools/create_segmentation_masks.py /workspace/data/dem_tiles_pits_1m/ /workspace/code/data/Hunting_pits_covered_by_lidar.shp object_id /workspace/data/object_detection/segmentation_masks_tiles_1m/

**Split segmentation masks into chips**

    python /workspace/code/tools/split_training_data.py /workspace/data/object_detection/segmentation_masks_tiles_05m/ /workspace/data/object_detection/split_segmentations_masks_05m/ --tile_size 250
    python /workspace/code/tools/split_training_data.py /workspace/data/object_detection/segmentation_masks_tiles_1m/ /workspace/data/object_detection/split_segmentations_masks_1m/ --tile_size 250

**Convert selected segmentation masks to bounding boxes**

Use the object detection docker image to create bounding boxes.

    docker run -it --gpus all -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar detection:latest

    python /workspace/code/object_detection/masks_to_boxes.py /workspace/temp/ /workspace/data/object_detection/split_segmentations_masks_05m/ 250 0 /workspace/data/object_detection/bounding_boxes_05m/


failed at 18006.tif, 26039.tif, 


    python /workspace/code/object_detection/masks_to_boxes.py /workspace/temp/ /workspace/data/object_detection/split_segmentations_masks_1m/ 250 0 /workspace/data/object_detection/bounding_boxes_1m/

failed at 6214.tif 19707.tif    20079.tif






**copy image chips and bounding boxes to the final directory**
    python /workspace/code/tools/copy_correct_chips.py /workspace/data/split_data_pits_05m/  /workspace/data/object_detection/bounding_boxes_05m/ /workspace/data/final_data_05m/training/

    python /workspace/code/tools/copy_correct_chips.py /workspace/data/split_data_pits_1m/  /workspace/data/object_detection/bounding_boxes_1m/ /workspace/data/final_data_1m/training/


**Create data split and move test data to new directories**
create data split between training and testing using this script. The batch script partition_data.sh cleans the test data directories and moves the test chips to respective test directory using a 80% vs 20% train / test split. Run it with:
    
    ./partition_data_05m.sh
    ./partition_data_1m.sh

    python /workspace/code/object_detection/YoloBBoxChecker/main.py


# Train and evaluate Unet


The training and evaluation of test chips can be done with these batch scripts:

        ./Pre_train_UNets_with_the_moon.sh
        ./train_test_unet_05m.sh
        ./train_test_unet_1m.sh
        ./train_test_xception_unet_05m.sh
        ./train_test_xception_unet_1m.sh



**Demo area**
Extrat dems
    python /workspace/code/tools/laz_to_dem.py /workspace/data/demo_area/tiles/ /workspace/data/demo_area/dem_tiles/ 0.5

    python /workspace/code/tools/laz_to_dem.py /workspace/data/demo_area/tiles/ /workspace/data/demo_area/dem_tiles_1m/ 1.0
    python /workspace/code/tools/laz_to_dem.py /workspace/data/demo_area/tiles/ /workspace/data/demo_area/dem_tiles_test/ 1.0

Calculate topoindicies

    python /workspace/code/Extract_topographcical_indices_05m.py /workspace/temp/ /workspace/data/demo_area/dem_tiles/ /workspace/data/demo_area/topographical_indicies_05m/hillshade/ /workspace/data/demo_area/topographical_indicies_05m/maxelevationdeviation/ /workspace/data/demo_area/topographical_indicies_05m/multiscaleelevationpercentile/ /workspace/data/demo_area/topographical_indicies_05m/minimal_curvature/ /workspace/data/demo_area/topographical_indicies_05m/maximal_curvature/ /workspace/data/demo_area/topographical_indicies_05m/profile_curvature/ /workspace/data/demo_area/topographical_indicies_05m/stdon/ /workspace/data/demo_area/topographical_indicies_05m/multiscale_stdon/ /workspace/data/demo_area/topographical_indicies_05m/elevation_above_pit/ /workspace/data/demo_area/topographical_indicies_05m/depthinsink/

    python /workspace/code/Extract_topographcical_indices_1m.py /workspace/temp/ /workspace/data/demo_area/dem_tiles_1m/ /workspace/data/demo_area/topographical_indicies_1m/hillshade/ /workspace/data/demo_area/topographical_indicies_1m/maxelevationdeviation/ /workspace/data/demo_area/topographical_indicies_1m/multiscaleelevationpercentile/ /workspace/data/demo_area/topographical_indicies_1m/minimal_curvature/ /workspace/data/demo_area/topographical_indicies_1m/maximal_curvature/ /workspace/data/demo_area/topographical_indicies_1m/profile_curvature/ /workspace/data/demo_area/topographical_indicies_1m/stdon/ /workspace/data/demo_area/topographical_indicies_1m/multiscale_stdon/ /workspace/data/demo_area/topographical_indicies_1m/elevation_above_pit/ /workspace/data/demo_area/topographical_indicies_1m/depthinsink/

**Inference using the best indices**


    python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/demo_area/topographical_indicies_05m/minimal_curvature /workspace/data/logfiles/UNet/05m/minimal_curvature1/trained.h5 /workspace/data/demo_area/topographical_indicies_05m/inference/ UNet --classes 0,1
    python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/demo_area/topographical_indicies_1m/minimal_curvature /workspace/data/logfiles/UNet/1m/minimal_curvature1/trained.h5 /workspace/data/demo_area/topographical_indicies_1m/inference/ UNet --classes 0,1

    python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/demo_area/topographical_indicies_05m/maximal_curvature /workspace/data/logfiles/ExceptionUNet/05m/maximal_curvature1/trained.h5 /workspace/data/demo_area/topographical_indicies_05m/inference_exception/ XceptionUNet --classes 0,1
    python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/demo_area/topographical_indicies_1m/profile_curvature /workspace/data/logfiles/ExceptionUNet/1m/profile_curvature1/trained.h5 /workspace/data/demo_area/topographical_indicies_1m/inference_exception/ XceptionUNet --classes 0,1    



    python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/demo_area/topographical_indicies_05m/inference_exception/ /workspace/data/demo_area/topographical_indicies_05m/inference_exception_post_processed/ --output_type=polygon --min_area=30 --min_ratio=-1
    python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/demo_area/topographical_indicies_1m/inference_exception/ /workspace/data/demo_area/topographical_indicies_1m/inference_exception_post_processed/ --output_type=polygon --min_area=30 --min_ratio=-1



**convert test chips to polygon**

    python /workspace/code/semantic_segmentation/post_processing.py /workspace/temp/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/final_data_05m/testing/polygon_labels/ --output_type=polygon --min_area=5 --min_ratio=-0.5











# Object detection
This section uses the docker container tagged "detection". the segmentation container is used for some steps since I have not figured out how to install gdal in the detection container.   

    screen -S detection


    docker run -it --gpus all -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest




**Partition training data**\
 The data is split into training data, testing data and validation data. The traning data will be used to train the model and validation data will be used to experiment while the testing data is held for the final results. The datasplit was already done during the segmentation and will be reused for the object detection.
 

    



# Transfer learning

![alt text](images/Crater.png)

## Data description
Impact creaters from the moon were used to pre-train the model. These creaters were digitised by NASA and are avalible from the Moon Crater Database v1 Robbins:https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/lunar_crater_database_robbins_2018 The database contains approximately 1.3 million lunar impact craters and is approximately complete for all craters larger than about 1–2 km in diameter. Craters were manually identified and measured on Lunar Reconnaissance Orbiter (LRO) Camera (LROC) Wide-Angle Camera (WAC) images, in LRO Lunar Orbiter Laser Altimeter (LOLA) topography, SELENE Kaguya Terrain Camera (TC) images, and a merged LOLA+TC DTM.


The Moon LRO LOLA DEM 118m v1 was used as digital elevation model. This digital elevation model  is based on data from the Lunar Orbiter Laser Altimeter, an instrument on the National Aeronautics and Space Agency (NASA) Lunar Reconnaissance Orbiter (LRO) spacecraft. The created DEM represents more than 6.5 billion measurements gathered between July 2009 and July 2013, adjusted for consistency in the coordinate system described below, and then converted to lunar radii.
Source: https://astrogeology.usgs.gov/search/details/Moon/LRO/LOLA/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014/cub

## Select craters
The charcoal kilns in the trainig data were between x and y pixels with an average of z in diameter. Therefore creaters that were less than x pixels (of the lundar dem) were excluded. creaters larger than y pixels were resampled down to z pixels. The following criterias:

    1. They can not overlap any nearby creaters.
    2. They have to be about the same size range as charcoal kilns in number of pixels.
    3. min and max lat and log values to avoid deformed craters?



## Evaluate model
    python Y:/William/GitHub/Remnants-of-charcoal-kilns/evaluate_model.py Y:/William/Kolbottnar/data/selected_data/hillshade/ Y:/William/Kolbottnar/data/selected_data/labels/ Y:/William/Kolbottnar/logs/log43/valid_imgs.txt Y:/William/Kolbottnar/logs/log43/test.h5 Y:/William/Kolbottnar/logs/log43/evaluation.csv --wo_crf

## Run inference
    python Y:/William/GitHub/Remnants-of-charcoal-kilns/inference.py Y:/William/Kolbottnar/data/topographical_indices/hillshade/ Y:/William/Kolbottnar/logs/log34/test.h5 D:/kolbottnar/inference/34_inference/ --tile_size=256 --wo_crf

## postprocessing
**min_area** is extracted from the smallest kiln in the training data.  
**min_ratio** is the perimeter to area ratio of the vector polygons. -0.3 is based on the training data.  

    python Y:/William/GitHub/Remnants-of-charcoal-kilns/post_processing.py D:/kolbottnar/inference/34_inference/ D:/kolbottnar/inference/34_post_processing/raw_polygons/ D:/kolbottnar/inference/34_post_processing/filtered_polygons/ --min_area=400 --min_ratio=-0.3
 

# Transfer learning

## The moon

The Lunar creaters were then used to create a binary raster mask where 1 is a creater and 0 is background. Creaters between x and x latidude were selected to avoid distorted creaters near the poles. Two segmentation masks were created. 1 for segmentation and 1 for object detection. source: https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2018JE005592 
    
    python /workspace/code/tools/create_segmentation_masks.py /workspace/data/lunar_data/dem_lat_50/ /workspace/data/lunar_data/Catalog_Moon_Release_20180815_shapefile180/Catalog_Moon_Release_20180815_1kmPlus_180.shp Classvalue /workspace/data/lunar_data/topographical_indices_normalized/labels/

    python /workspace/code/tools/create_segmentation_masks.py /workspace/data/lunar_data/dem_lat_50/ /workspace/data/lunar_data/Catalog_Moon_Release_20180815_shapefile180/Catalog_Moon_Release_20180815_1kmPlus_180.shp FID /workspace/data/lunar_data/topographical_indices_normalized/object_labels/


**Extract topographical indices**

    python /workspace/code/Extract_topographcical_indices_05m.py /workspace/temp/ /workspace/data/lunar_data/dem_lat_50/ /workspace/data/lunar_data/topographical_indices_normalized/hillshade/ /workspace/data/lunar_data/topographical_indices_normalized/maxelevationdeviation/ /workspace/data/lunar_data/topographical_indices_normalized/multiscaleelevationpercentile/ /workspace/data/lunar_data/topographical_indices_normalized/minimal_curvature/ /workspace/data/lunar_data/topographical_indices_normalized/maximal_curvature/ /workspace/data/lunar_data/topographical_indices_normalized/profile_curvature/ /workspace/data/lunar_data/topographical_indices_normalized/stdon/ /workspace/data/lunar_data/topographical_indices_normalized/multiscale_stdon/ /workspace/data/lunar_data/topographical_indices_normalized/elevation_above_pits/ /workspace/data/lunar_data/topographical_indices_normalized/depthinsink/

**Split lunar data into 250x250 chips**

    ./workspace/code/split_indicies_moon.sh


example Split minimal_curvature

python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/minimal_curvature/ /workspace/data/lunar_data/split_data/minimal_curvature/ --tile_size 250
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/labels/ /workspace/data/lunar_data/split_data/labels/ --tile_size 250
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/hillshade/ /workspace/data/lunar_data/split_data/hillshade/ --tile_size 250


**Convert selected segmentation masks to bounding boxes**

Use the object detection docker image to create bounding boxes.

    docker run -it --gpus all -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar detection:latest bash

    python /workspace/code/object_detection/masks_to_boxes.py /workspace/temp/ /workspace/data/lunar_data/split_data/object_labels/ 250 0 /workspace/data/lunar_data/bounding_boxes/

    python /workspace/code/tools/copy_correct_chips.py /workspace/data/lunar_data/split_data/ /workspace/data/lunar_data/bounding_boxes/ /workspace/data/lunar_data/final_data/training/


    ./code/partition_data_moon.sh

## Train a u-net on the moon
-p 8888:8888 --mount type=bind,src=G:\moon\code,target=/workspace/code --mount type=bind,src=G:\moon\data,target=/workspace/data segmentation:latest
    docker run -it --gpus all --mount type=bind,src=G:\moon\code,target=/workspace/code --mount type=bind,src=G:\moon\data,target=/workspace/data segmentation:latest




    python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/lunar_data/final_data/training/maximal_curvature/ /workspace/data/lunar_data/final_data/training/labels/ /workspace/data/logfiles/moon/ --weighting="mfb" --depth=4 --epochs=100 --batch_size=128 --classes=0,1
    python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_05m/testing/maximal_curvature/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/moon/trained.h5 /workspace/data/logfiles/moon/test.csv --classes=0,1 --depth=4

    ./workspace/code/semantic_segmentation/train_test_unet_moon.sh

# CA-NET
Comprehensive Attention Convolutional Neural Networks


## contact information
Mail:
<William.lidberg@slu.se>\
Phone:
+46706295567\
[Twitter](https://twitter.com/william_lidberg)
## License
This project is licensed under the MIT License - see the LICENSE file for details.

# References
