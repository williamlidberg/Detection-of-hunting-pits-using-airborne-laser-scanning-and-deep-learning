# Remnants-of-charcoal-kilns
Detect remnants of charcoal kilns from LiDAR data

![alt text](BlackWhite_large_zoom_wide2.png)

## Anaconda -python 3.8.12  
**Not sure whats going on with h5py, will fix in container later**  
conda install cudatoolkit=11.3.1 -c conda-forge -y
conda install -c conda-forge cudnn=8.2.1 -y  
pip install tensorflow==2.5.0
pip install whitebox==2.0.3   
conda install -c conda-forge opencv -y  
conda install -c conda-forge tifffile -y  
conda install -c anaconda pandas -y  
conda install -c anaconda scikit-learn -y  
conda install -c conda-forge gdal -y  
conda install geopandas - y
pip install splitraster  

pip uninstall h5py   
pip install h5py # had to reinstall for the training to work  

conda install h5py -y # had to use conda for inference to work  
conda uninstall h5py -y # honestly dont remember. will fix later.  

## Select 0.5 m dem tiles based on locatiaon of training data  
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Select_study_areas.py D:/kolbottnar/Kolbottnar.shp Y:/William/Kolbottnar/data/footprint/Footprint.shp F:/DitchNet/HalfMeterData/dem05m/ Y:/William/Kolbottnar/data/selected_dems/

## Convert field observations to labeled tiles  
python Y:/William/GitHub/Remnants-of-charcoal-kilns/create_labels.py Y:/William/Kolbottnar/data/selected_dems/ Y:/William/GitHub/Remnants-of-charcoal-kilns/data/charcoal_kilns_buffer.shp Y:/William/Kolbottnar/data/label_tiles/

## Topographical indices  
**Extract topographical indices from dem tiles**  
python Y:/William/GitHub/Remnants-of-charcoal-kilns/Topographical_indicies.py Y:/William/Kolbottnar/data/selected_dems/ Y:/William/Kolbottnar/data/topographical_indices/hillshade/ Y:/William/Kolbottnar/data/topographical_indices/slope/ Y:/William/Kolbottnar/data/topographical_indices/hpmf/

**normalize topographical indices**  
python Y:/William/GitHub/Remnants-of-charcoal-kilns/normalize_indices.py Y:/William/Kolbottnar/data/topographical_indices/hillshade/ Y:/William/Kolbottnar/data/topographical_indices_normalized/hillshade/ Y:/William/Kolbottnar/data/topographical_indices/slope/ Y:/William/Kolbottnar/data/topographical_indices_normalized/slope/ Y:/William/Kolbottnar/data/topographical_indices/hpmf/ Y:/William/Kolbottnar/data/topographical_indices_normalized/hpmf/


## Split tiles into smaller image chips make sure the directory is empty/new so the split starts at 1 each time.  
**Split hillshade**  
python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/topographical_indices_normalized/hillshade/ Y:/William/Kolbottnar/data/split_data/hillshade/ --tile_size 256

**Split slope**  
python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/topographical_indices_normalized/slope/ Y:/William/Kolbottnar/data/split_data/slope/ --tile_size 256

**split high pass median filter**  
python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/topographical_indices_normalized/hpmf/ Y:/William/Kolbottnar/data/split_data/hpmf/ --tile_size 256

**Split labels**   
python Y:/William/GitHub/Remnants-of-charcoal-kilns/split_training_data.py Y:/William/Kolbottnar/data/label_tiles/ Y:/William/Kolbottnar/data/split_data/labels/ --tile_size 256

## Select chips with labeled pixels (This could be reworked to delete no label chips instead of copying chips with labels)
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
log 42 was trained on normalized hillshade images. average mcc0.72 total mcc 0.84  
log44 was trained on normalised slope  

**Set up tensorboard**  
tensorboard --logdir Y:/William/Kolbottnar/logs/log42

**note to self: investigate if chips are overlapping and causing overfitting issues**  

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

python Y:/William/GitHub/Remnants-of-charcoal-kilns/post_processing.py R:/Temp/inference2/ R:/Temp/post_processing/raw_polygons/ R:/Temp/post_processing/filtered_polygons/ --min_area=400 --min_ratio=-0.3

## Docker - needs an update
Build container  

docker build -t charcoal .  

Run container connected to the NAS  

docker run --gpus all --shm-size=48g -it --mount type=bind,source=/mnt/nas1_extension_100tb/William/,target=/app charcoal:latest   


## License
This project is licensed under the MIT License - see the LICENSE file for details.
