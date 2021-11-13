# Remnants-of-charcoal-kilns
Detect remnants of charcoal kilns from LiDAR data

![alt text](BlackWhite_large_zoom_wide2.png)


## Select 0.5 m dem tiles based on locatiaon of training data
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Select_study_areas.py D:/kolbottnar/Kolbottnar.shp Y:/William/Kolbottnar/data/footprint/Footprint.shp F:/DitchNet/HalfMeterData/dem05m/ Y:/William/Kolbottnar/data/selected_dems/

## Convert field observations to label tiles
python Y:/William/GitHub/Remnants-of-charcoal-kilns/create_labels.py Y:/William/Kolbottnar/data/selected_dems/  D:/kolbottnar/Kolbottnar_buf.shp Y:/William/Kolbottnar/data/label_tiles/

## Extract topographical indices from dem tiles
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Topographical_indicies.py Y:/William/Kolbottnar/data/selected_dems/ Y:/William/Kolbottnar/data/topographical_indicies/hillshade/ Y:/William/Kolbottnar/data/topographical_indicies/slope/ Y:/William/Kolbottnar/data/topographical_indicies/hpmf/

## Split data into chips
python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/topographical_indicies/hillshade R:/Temp/split_tile --tile_size 500

python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/label_tiles R:/Temp/split_labels --tile_size 500

## Select chips with labeled pixels
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Select_chips_with_labels.py R:/Temp/split_tile/ R:/Temp/split_labels/ R:/Temp/selected_chips/images/ 1 R:/Temp/selected_chips/labels/

Build container

docker build -t charcoal .

Run container connected to the NAS

docker run --gpus all --shm-size=48g -it --mount type=bind,source=/mnt/nas1_extension_100tb/William/,target=/app charcoal:latest

Train with multiple bands by 

python train.py train/gt/ log/ XceptionUNet -I train/hpmf/ -I train/skyview/ --epochs 100 --steps_per_epoch 100
