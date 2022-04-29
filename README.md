# Remnants-of-charcoal-kilns
Detect remnants of charcoal kilns from LiDAR data

![alt text](BlackWhite_large_zoom_wide2.png)

# Docker
**Set up a docker container with a ramdisk**
docker build -t cultural .


**Run notebook**
screen -S notebook

docker run -it -p 8888:8888 --gpus all --mount type=bind,source=/mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns/,target=/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog/dem05m:/workspace/lidar cultural:latest

jupyter notebook --ip=0.0.0.0 --no-browser --allow-root

## Select 0.5 m dem tiles based on locatiaon of training data
**needs new dem**  
python /workspace/code/Remnants-of-charcoal-kilns/Select_study_areas.py workspace/data/charcoal_kilns/Merged_charcoal_kilns_william.shp /workspace/data/Footprint.shp /workspace/lidar/ /workspace/data/selected_dems/

## Convert field observations to labeled tiles  
python /code/create_labels.py /data/selected_dems/ /data/charcoal_kilns/charcoal_kilns_buffer.shp /data/label_tiles/

python /code/Topographical_indicies.py //data/selected_dems/ /data/topographical_indices/hillshade/ /data/topographical_indices/slope/ /data/topographical_indices/hpmf/

**normalize topographical indices**  
python /code/normalize_indices.py /data/topographical_indices/hillshade/ /data/topographical_indices_normalized/hillshade/ /data/topographical_indices/slope/ /data/topographical_indices_normalized/slope/ /data/topographical_indices/hpmf/ /data/topographical_indices_normalized/hpmf/

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

## Train the model
This is an example on how to train the model in the docker cotnainer:

python /workspace/code/train.py -I /workspace/data/split_data/hillshade/ -I /workspace/data/split_data/slope/ -I /workspace/data/split_data/hpmf/ /workspace/data/split_data/labels/ /workspace/data/logfiles/log1/ --seed=40 --epochs 10 



**Set up tensorboard**  
tensorboard --logdir Y:/William/Kolbottnar/logs/log42


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
