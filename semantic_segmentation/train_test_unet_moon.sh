#!/bin/bash 


#python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/lunar_data/final_data/training/maximal_curvature/ /workspace/data/lunar_data/final_data/training/labels/ /workspace/data/logfiles/moon/maximal_curvature/ --weighting="mfb" --depth=4 --epochs=100 --batch_size=64 --classes=0,1
#python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_05m/testing/maximal_curvature/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/moon/maximal_curvature/trained.h5 /workspace/data/logfiles/moon/maximal_curvature/test.csv --classes=0,1 --depth=4


python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/lunar_data/final_data/training/maxelevationdeviation/ /workspace/data/lunar_data/final_data/training/labels/ /workspace/data/logfiles/moon/maxelevationdeviation/ --weighting="mfb" --depth=4 --epochs=100 --batch_size=64 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_05m/testing/maxelevationdeviation/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/moon/maxelevationdeviation/trained.h5 /workspace/data/logfiles/moon/maxelevationdeviation/test.csv --classes=0,1 --depth=4



python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/lunar_data/final_data/training/minimal_curvature/ /workspace/data/lunar_data/final_data/training/labels/ /workspace/data/logfiles/moon/minimal_curvature/ --weighting="mfb" --depth=4 --epochs=100 --batch_size=64 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_05m/testing/minimal_curvature/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/moon/minimal_curvature/trained.h5 /workspace/data/logfiles/moon/minimal_curvature/test.csv --classes=0,1 --depth=4


python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/lunar_data/final_data/training/profile_curvature/ /workspace/data/lunar_data/final_data/training/labels/ /workspace/data/logfiles/moon/profile_curvature/ --weighting="mfb" --depth=4 --epochs=100 --batch_size=64 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_05m/testing/profile_curvature/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/moon/profile_curvature/trained.h5 /workspace/data/logfiles/moon/profile_curvature/test.csv --classes=0,1 --depth=4

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/lunar_data/final_data/training/elevation_above_pit/ /workspace/data/lunar_data/final_data/training/labels/ /workspace/data/logfiles/moon/elevation_above_pit/ --weighting="mfb" --depth=4 --epochs=100 --batch_size=64 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_05m/testing/elevation_above_pit/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/moon/elevation_above_pit/trained.h5 /workspace/data/logfiles/moon/elevation_above_pit/test.csv --classes=0,1 --depth=4

