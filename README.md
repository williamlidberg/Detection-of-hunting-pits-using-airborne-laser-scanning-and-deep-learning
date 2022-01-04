# Remnants-of-charcoal-kilns
Detect remnants of charcoal kilns from LiDAR data

![alt text](BlackWhite_large_zoom_wide2.png)

## Anaconeda -python 3.8.12
pip install tensorflow==2.5.0
pip install whitebox==2.0.3
conda install cudatoolkit=11.3.1 -c conda-forge -y
conda install -c conda-forge cudnn=8.2.1 -y
conda install -c conda-forge opencv -y
conda install -c conda-forge tifffile -y
conda install -c anaconda pandas -y
conda install -c anaconda scikit-learn -y
conda install -c conda-forge gdal -y
pip uninstall h5py 
pip install h5py
conda install geopandas -y

## Select 0.5 m dem tiles based on locatiaon of training data
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Select_study_areas.py D:/kolbottnar/Kolbottnar.shp Y:/William/Kolbottnar/data/footprint/Footprint.shp F:/DitchNet/HalfMeterData/dem05m/ Y:/William/Kolbottnar/data/selected_dems/

## Convert field observations to label tiles
python Y:/William/GitHub/Remnants-of-charcoal-kilns/create_labels.py Y:/William/Kolbottnar/data/selected_dems/  D:/kolbottnar/Kolbottnar_buf.shp Y:/William/Kolbottnar/data/label_tiles/

## Extract topographical indices from dem tiles
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Topographical_indicies.py Y:/William/Kolbottnar/data/selected_dems/ Y:/William/Kolbottnar/data/topographical_indicies/hillshade/ Y:/William/Kolbottnar/data/topographical_indicies/slope/ Y:/William/Kolbottnar/data/topographical_indicies/hpmf/

## Split data into chips
python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/topographical_indicies/hillshade R:/Temp/split_hillshade --tile_size 256

python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/label_tiles R:/Temp/split_labels --tile_size 256

## Select chips with labeled pixels
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Select_chips_with_labels.py R:/Temp/split_hillshade/ R:/Temp/split_labels/ R:/Temp/selected_chips/images/ 1 R:/Temp/selected_chips/labels/

## tensorboard
tensorboard --logdir Y:/William/Kolbottnar/logs/log33

## Train model
python Y:/William/GitHub/Remnants-of-charcoal-kilns/train.py R:/Temp/selected_chips/images/ R:/Temp/selected_chips/labels/ Y:/William/Kolbottnar/logs/log33 XceptionUNet --seed=40 

## Evaluate model
python Y:/William/GitHub/Remnants-of-charcoal-kilns/evaluate_model.py R:/Temp/selected_chips/images/ R:/Temp/selected_chips/labels/ Y:/William/Kolbottnar/logs/log33/valid_imgs.txt Y:/William/Kolbottnar/logs/log33/test.h5 Y:/William/Kolbottnar/logs/log33/evaluation.csv --wo_crf

## Run inference
python Y:/William/GitHub/Remnants-of-charcoal-kilns/inference.py Y:/William/Kolbottnar/data/topographical_indicies/hillshade/ Y:/William/Kolbottnar/logs/log33/test.h5 R:/Temp/inference/ --tile_size=256 --wo_crf

## postprocessing
python Y:/William/GitHub/Remnants-of-charcoal-kilns/post_processing.py R:/Temp/inference/ R:/Temp/post_processing/raw_polygons/ R:/Temp/post_processing/filtered_polygons/ --min_area=400 --min_ratio=-0.3

## Docker
Build container

docker build -t charcoal .

Run container connected to the NAS

docker run --gpus all --shm-size=48g -it --mount type=bind,source=/mnt/nas1_extension_100tb/William/,target=/app charcoal:latest

Train with multiple bands by 

python train.py train/gt/ log/ XceptionUNet -I train/hpmf/ -I train/skyview/ --epochs 100 --steps_per_epoch 100
