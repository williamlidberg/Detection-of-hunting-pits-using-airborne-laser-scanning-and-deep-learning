# Remnants-of-charcoal-kilns
Detect remnants of charcoal kilns from LiDAR data

![alt text](BlackWhite_large_zoom_wide2.png)


## Select 0.5 m dem tiles based on locatiaon of training data
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Select_study_areas.py D:/kolbottnar/Kolbottnar.shp Y:/William/Kolbottnar/data/footprint/Footprint.shp F:/DitchNet/HalfMeterData/dem05m/ Y:/William/Kolbottnar/data/selected_dems/

## Extract topographical indices from dem tiles
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Topographical_indicies.py Y:/William/Kolbottnar/data/selected_dems/ Y:/William/Kolbottnar/data/topographical_indicies/hillshade/ Y:/William/Kolbottnar/data/topographical_indicies/slope/ Y:/William/Kolbottnar/data/topographical_indicies/hpmf/

## Split data into chips
python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/topographical_indicies/hillshade R:/Temp/split_tile --tile_size 500

Build container

docker build -t charcoal .

Run container connected to the NAS

docker run --gpus all --shm-size=48g -it --mount type=bind,source=/mnt/nas1_extension_100tb/William/,target=/app charcoal:latest

Train with multiple bands by 

python train.py train/gt/ log/ XceptionUNet -I train/hpmf/ -I train/skyview/ --epochs 100 --steps_per_epoch 100
