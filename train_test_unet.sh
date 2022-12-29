#!/bin/bash 
echo "train unet"
docker run -it --gpus all -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/hillshade/ -I /workspace/data/split_data_pits/elevation_above_pit/ -I /workspace/data/split_data_pits/stdon/ -I /workspace/data/split_data_pits/minimal_curvature/ -I /workspace/data/split_data_pits/profile_curvature/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/pits6/ --weighting="0.005,1" --seed=40 --epochs 100
echo "evaluate unet on eval data"
docker run -it --gpus all -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/hillshade/ -I /workspace/data/split_data_pits/elevation_above_pit/ -I /workspace/data/split_data_pits/stdon/ -I /workspace/data/split_data_pits/minimal_curvature/ -I /workspace/data/split_data_pits/profile_curvature/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/pits6/trained.h5 /workspace/data/logfiles/pits/pits6/eval.csv --selected_imgs=/workspace/data/logfiles/pits/pits6/valid_imgs.txt --classes=0,1 
echo "inference on test tile"
docker run -it --gpus all -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/semantic_segmentation/inference_unet.py -I /workspace/data/test_data_pits/hillshade/0227.tif -I /workspace/data/test_data_pits/elevation_above_pit/0227.tif -I /workspace/data/test_data_pits/stdon/0227.tif -I /workspace/data/test_data_pits/minimal_curvature/0227.tif -I /workspace/data/test_data_pits/profile_curvature/0227.tif /workspace/data/logfiles/pits/pits6/trained.h5 /workspace/data/logfiles/pits/pits6/