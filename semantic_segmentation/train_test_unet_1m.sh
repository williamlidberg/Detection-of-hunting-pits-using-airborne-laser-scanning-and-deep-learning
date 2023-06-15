#!/bin/bash 
echo "hillshade"
mkdir /workspace/data/logfiles/UNet/1m/hillshade1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_1m/training/hillshade/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/UNet/1m/hillshade1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/UNet/1m/hillshade1/trained.h5 UNet /workspace/data/logfiles/UNet/1m/hillshade1/test.csv --classes=0,1


echo "maxelevationdeviation"
mkdir /workspace/data/logfiles/UNet/1m/maxelevationdeviation1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_1m/training/maxelevationdeviation/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/UNet/1m/maxelevationdeviation1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/UNet/1m/maxelevationdeviation1/trained.h5 UNet /workspace/data/logfiles/UNet/1m/maxelevationdeviation1/test.csv --classes=0,1


echo "multiscaleelevationpercentile"
mkdir /workspace/data/logfiles/UNet/1m/multiscaleelevationpercentile1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/UNet/1m/multiscaleelevationpercentile1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/UNet/1m/multiscaleelevationpercentile1/trained.h5 UNet /workspace/data/logfiles/UNet/1m/multiscaleelevationpercentile1/test.csv --classes=0,1

echo "training/minimal_curvature"
mkdir /workspace/data/logfiles/UNet/1m/minimal_curvature1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_1m/training/minimal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/UNet/1m/minimal_curvature1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_1m/testing/minimal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/UNet/1m/minimal_curvature1/trained.h5 UNet /workspace/data/logfiles/UNet/1m/minimal_curvature1/test.csv --classes=0,1

echo "maximal_curvature"
mkdir /workspace/data/logfiles/UNet/1m/maximal_curvature1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_1m/training/maximal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/UNet/1m/maximal_curvature1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_1m/testing/maximal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/UNet/1m/maximal_curvature1/trained.h5 UNet /workspace/data/logfiles/UNet/1m/maximal_curvature1/test.csv --classes=0,1

echo "profile_curvature"
mkdir /workspace/data/logfiles/UNet/1m/profile_curvature1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_1m/training/profile_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/UNet/1m/profile_curvature1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_1m/testing/profile_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/UNet/1m/profile_curvature1/trained.h5 UNet /workspace/data/logfiles/UNet/1m/profile_curvature1/test.csv --classes=0,1


echo "stdon"
mkdir /workspace/data/logfiles/UNet/1m/stdon1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_1m/training/stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/UNet/1m/stdon1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_1m/testing/stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/UNet/1m/stdon1/trained.h5 UNet /workspace/data/logfiles/UNet/1m/stdon1/test.csv --classes=0,1

echo "multiscale_stdon"
mkdir /workspace/data/logfiles/UNet/1m/multiscale_stdon1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_1m/training/multiscale_stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/UNet/1m/multiscale_stdon1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/UNet/1m/multiscale_stdon1/trained.h5 UNet /workspace/data/logfiles/UNet/1m/multiscale_stdon1/test.csv --classes=0,1


echo "elevation_above_pit"
mkdir /workspace/data/logfiles/UNet/1m/elevation_above_pit1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_1m/training/elevation_above_pit/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/UNet/1m/elevation_above_pit1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_1m/testing/elevation_above_pit/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/UNet/1m/elevation_above_pit1/trained.h5 UNet /workspace/data/logfiles/UNet/1m/elevation_above_pit1/test.csv --classes=0,1



echo "depthinsink"
mkdir /workspace/data/logfiles/UNet/1m/depthinsink1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/UNet/1m/depthinsink1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/UNet/1m/depthinsink1/trained.h5 UNet /workspace/data/logfiles/UNet/1m/depthinsink1/test.csv --classes=0,1



