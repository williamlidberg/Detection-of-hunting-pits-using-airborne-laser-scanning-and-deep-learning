[![docs](https://img.shields.io/badge/whitebox-docs-brightgreen.svg)](https://www.whiteboxgeo.com/manual/wbt_book/preface.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Twitter Follow](https://img.shields.io/twitter/follow/William_Lidberg?style=social)](https://twitter.com/giswqs)



# Remnants-of-charcoal-kilns
Detect remnants of charcoal kilns from LiDAR data

![alt text](BlackWhite_large_zoom_wide2.png)



**Contents**

1. [Set up docker](#Docker)
2. [Create digital elevation model](#Create digital elevation model)
3. [Create training data](#Create training data)
4. [Train the model](#Train the model)
5. [The Moon](#The Moon)


# Docker
**Set up a docker container with a ramdisk**

Navigate to /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns/ and run:
docker build -t cultural .


**Run notebook**
screen -S notebook

docker run -it -p 8888:8888 -p 16006:16006 --gpus all --mount type=bind,source=/mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns/,target=/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar cultural:latest

jupyter notebook --ip=0.0.0.0 --no-browser --allow-root

# Create digital elevation model

## Select lidar tiles based on locatiaon of training data
First create a polygon that can be used to select relevant laz tiles:\
python /workspace/code/create_aoi_poolygon.py /workspace/lidar/none.shp /workspace/data/hunting_pits/Fangstgrop_training_Holmen_Cissi_695st_220214.shp /workspace/lidar/pooled_laz_files/ /workspace/data/hunting_pits/laz/

## Convert laz to dem
python /workspace/code/laz_to_dem.py /workspace/data/hunting_pits/laz/ /workspace/data/hunting_pits/dem_tiles/

# Create training data
## Convert field observations to labeled tiles  
python /code//utils/create_labels.py /data/selected_dems/ /data/charcoal_kilns/charcoal_kilns_buffer.shp /data/label_tiles/

## Extract and normalize topographical indices
python /workspace/code/Extract_topographcical_indices.py /workspace/temp_dir/ /workspace/data/hunting_pits/laz/ /workspace/data/hunting_pits/topographical_indices_normalized/hillshade/ /workspace/data/hunting_pits/topographical_indices_normalized/slope/ /workspace/data/hunting_pits/topographical_indices_normalized/hpmf/ /workspace/data/hunting_pits/topographical_indices_normalized/stdon/

## Split tiles into smaller image chips make sure the directory is empty/new so the split starts at 1 each time. 
Use this to empty the split directories before running the split scripts: rm -f /workspace/data/split_data/{*,.*}
 
**Split hillshade**  
python /code/split_training_data.py /data/topographical_indices_normalized/hillshade/ /data/split_data/hillshade/ --tile_size 256

**Split slope**  
python /code/split_training_data.py /data/topographical_indices_normalized/slope/ /data/split_data/slope/ --tile_size 256

**split high pass median filter**  
python /code/split_training_data.py /data/topographical_indices_normalized/hpmf/ /data/split_data/hpmf/ --tile_size 256

**Split labels**   
python /code/split_training_data.py /data/label_tiles/ /data/split_data/labels/ --tile_size 256


**Remove chips without labels**
python /workspace/code/remove_unlabled_chips.py 1 /workspace/data/split_data/labels/ /workspace/data/split_data/hillshade/ /workspace/data/split_data/slope/ /workspace/data/split_data/hpmf/

# Train the model
This is an example on how to train the model in the docker cotnainer:

python /workspace/code/train.py -I /workspace/data/split_data/hillshade/ -I /workspace/data/split_data/slope/ -I /workspace/data/split_data/hpmf/ /workspace/data/split_data/labels/ /workspace/data/logfiles/log1/ --seed=40 --epochs 10 



# The Moon
## Data description
Impact creaters from the moon were used to pre-train the model. These creaters were digitised by NASA and are avalible from the Moon Crater Database v1 Robbins:https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/lunar_crater_database_robbins_2018 The database contains approximately 1.3 million lunar impact craters and is approximately complete for all craters larger than about 1–2 km in diameter. Craters were manually identified and measured on Lunar Reconnaissance Orbiter (LRO) Camera (LROC) Wide-Angle Camera (WAC) images, in LRO Lunar Orbiter Laser Altimeter (LOLA) topography, SELENE Kaguya Terrain Camera (TC) images, and a merged LOLA+TC DTM.


The Moon LRO LOLA DEM 118m v1 was used as digital elevation model. This digital elevation model  is based on data from the Lunar Orbiter Laser Altimeter, an instrument on the National Aeronautics and Space Agency (NASA) Lunar Reconnaissance Orbiter (LRO) spacecraft. The created DEM represents more than 6.5 billion measurements gathered between July 2009 and July 2013, adjusted for consistency in the coordinate system described below, and then converted to lunar radii.
Source: https://astrogeology.usgs.gov/search/details/Moon/LRO/LOLA/Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014/cub

## Crater selection
The charcoal kilns in the trainig data were between x and y pixels with an average of z in diameter. Therefore creaters that were less than x pixels (of the lundar dem) were excluded. creaters larger than y pixels were resampled down to z pixels. The following criterias:

    1. They can not overlap any nearby creaters.
    2. They have to be about the same size range as charcoal kilns in number of pixels.
    3. min and max lat and log values to avoid deformed craters?

    

# Old notes
## Train model - test with hillshade only  
log 34 was trained on only hillshades and got mcc 0.85  
log 35 was trained on only hpmf and got mcc 0  
log 36 was trained on only slope and got mcc 0  

hpmf and slope did not work. normalise?  
log 42 was trained on normalized hillshade images. average mcc0.72 total mcc 0.84  
log44 was trained on normalised slope  
**train the model without CRF**  
python Y:/William/GitHub/Remnants-of-charcoal-kilns/train.py Y:/William/Kolbottnar/data/selected_data/hillshade/ Y:/William/Kolbottnar/data/selected_data/labels/ Y:/William/Kolbottnar/logs/log42 XceptionUNet --seed=40 

hpmf  

python Y:/William/GitHub/Remnants-of-charcoal-kilns/train.py Y:/William/Kolbottnar/data/selected_data/hpmf/ Y:/William/Kolbottnar/data/selected_data/labels/ Y:/William/Kolbottnar/logs/log43 XceptionUNet --seed=40 

## Evaluate model
python Y:/William/GitHub/Remnants-of-charcoal-kilns/evaluate_model.py Y:/William/Kolbottnar/data/selected_data/hillshade/ Y:/William/Kolbottnar/data/selected_data/labels/ Y:/William/Kolbottnar/logs/log43/valid_imgs.txt Y:/William/Kolbottnar/logs/log43/test.h5 Y:/William/Kolbottnar/logs/log43/evaluation.csv --wo_crf

## Run inference
python Y:/William/GitHub/Remnants-of-charcoal-kilns/inference.py Y:/William/Kolbottnar/data/topographical_indices/hillshade/ Y:/William/Kolbottnar/logs/log34/test.h5 D:/kolbottnar/inference/34_inference/ --tile_size=256 --wo_crf

## postprocessing
**min_area** is extracted from the smallest kiln in the training data.  
**min_ratio** is the perimeter to area ratio of the vector polygons. -0.3 is based on the training data.  

python Y:/William/GitHub/Remnants-of-charcoal-kilns/post_processing.py D:/kolbottnar/inference/34_inference/ D:/kolbottnar/inference/34_post_processing/raw_polygons/ D:/kolbottnar/inference/34_post_processing/filtered_polygons/ --min_area=400 --min_ratio=-0.3
 


## License
This project is licensed under the MIT License - see the LICENSE file for details.
