#!/bin/bash 
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/lunar_data/final_data/training/hillshade/ /workspace/data/lunar_data/final_data/training/labels/ /workspace/data/logfiles/moon/unet UNet --seed=1 --epochs=25 --batch_size=32 --classes=0,1 --weighting="focal"
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/lunar_data/final_data/training/hillshade/ /workspace/data/lunar_data/final_data/training/labels/ /workspace/data/logfiles/moon/exception_unet XceptionUNet --seed=1 --epochs=25 --batch_size=32 --classes=0,1 --weighting="focal"
