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
pip install splitraster
conda install h5py -y
conda uninstall h5py -y
## Select 0.5 m dem tiles based on locatiaon of training data
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Select_study_areas.py D:/kolbottnar/Kolbottnar.shp Y:/William/Kolbottnar/data/footprint/Footprint.shp F:/DitchNet/HalfMeterData/dem05m/ Y:/William/Kolbottnar/data/selected_dems/

## Convert field observations to labeled tiles
python Y:/William/GitHub/Remnants-of-charcoal-kilns/create_labels.py Y:/William/Kolbottnar/data/selected_dems/ Y:/William/GitHub/Remnants-of-charcoal-kilns/data/charcoal_kilns_buffer.shp Y:/William/Kolbottnar/data/label_tiles/

## Extract topographical indices from dem tiles
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Topographical_indicies.py Y:/William/Kolbottnar/data/selected_dems/ Y:/William/Kolbottnar/data/topographical_indicies/hillshade/ Y:/William/Kolbottnar/data/topographical_indicies/slope/ Y:/William/Kolbottnar/data/topographical_indicies/hpmf/

## Split tiles into smaller image chips (ry to split tiles into 500 x 500 pixel chips to avoid overlap)
**Split hillshade**
python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/topographical_indicies/hillshade Y:/William/Kolbottnar/data/split_data/hillshade/ --tile_size 256

**Split slope**
python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/topographical_indicies/slope Y:/William/Kolbottnar/data/split_data/slope/ --tile_size 256

**split high pass median filter**
python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/topographical_indicies/hpmf Y:/William/Kolbottnar/data/split_data/hpmf/ --tile_size 256

**Split labels**
python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/label_tiles Y:/William/Kolbottnar/data/split_data/labels/ --tile_size 256

## Select chips with labeled pixels (This could be reworked to delete no label chips instead of copying hips with labels)
**Select hillshade**
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Select_chips_with_labels.py Y:/William/Kolbottnar/data/split_data/hillshade/ Y:/William/Kolbottnar/data/split_data/labels/ Y:/William/Kolbottnar/data/selected_data/hillshade/ 1 Y:/William/Kolbottnar/data/selected_data/labels/

**Select slope**
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Select_chips_with_labels.py Y:/William/Kolbottnar/data/split_data/slope/ Y:/William/Kolbottnar/data/split_data/labels/ Y:/William/Kolbottnar/data/selected_data/slope/ 1 Y:/William/Kolbottnar/data/selected_data/labels/

**Select high pass median filter**
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Select_chips_with_labels.py Y:/William/Kolbottnar/data/split_data/hpmf/ Y:/William/Kolbottnar/data/split_data/labels/ Y:/William/Kolbottnar/data/selected_data/hpmf/ 1 Y:/William/Kolbottnar/data/selected_data/labels/

## Train model - test with hillshade only
log 34 was trained on only hillshades and got mcc 0.85
log 35 was trained on only hpmf and got mcc 0
log 36 was trained on only slope and got mcc 0

hpmf and slope did not work. normalise?

**Set up tensorboard**
tensorboard --logdir Y:/William/Kolbottnar/logs/log36

**note to self: investigate if chips are overlapping and causing overfitting issues**
python Y:/William/GitHub/Remnants-of-charcoal-kilns/train.py Y:/William/Kolbottnar/data/selected_data/slope/ Y:/William/Kolbottnar/data/selected_data/labels/ Y:/William/Kolbottnar/logs/log36 XceptionUNet --seed=40 

## Evaluate model
python Y:/William/GitHub/Remnants-of-charcoal-kilns/evaluate_model.py Y:/William/Kolbottnar/data/selected_data/hillshade/ Y:/William/Kolbottnar/data/selected_data/labels/ Y:/William/Kolbottnar/logs/log36/valid_imgs.txt Y:/William/Kolbottnar/logs/log36/test.h5 Y:/William/Kolbottnar/logs/log36/evaluation.csv --wo_crf

## Run inference
python Y:/William/GitHub/Remnants-of-charcoal-kilns/inference.py Y:/William/Kolbottnar/data/topographical_indicies/hillshade/ Y:/William/Kolbottnar/logs/log34/test.h5 R:/Temp/inference2/ --tile_size=256 --wo_crf

## postprocessing
python Y:/William/GitHub/Remnants-of-charcoal-kilns/post_processing.py R:/Temp/inference2/ R:/Temp/post_processing/raw_polygons/ R:/Temp/post_processing/filtered_polygons/ --min_area=400 --min_ratio=-0.3

## Docker
Build container

docker build -t charcoal .

Run container connected to the NAS

docker run --gpus all --shm-size=48g -it --mount type=bind,source=/mnt/nas1_extension_100tb/William/,target=/app charcoal:latest

Train with multiple bands by 

python train.py train/gt/ log/ XceptionUNet -I train/hpmf/ -I train/skyview/ --epochs 100 --steps_per_epoch 100
