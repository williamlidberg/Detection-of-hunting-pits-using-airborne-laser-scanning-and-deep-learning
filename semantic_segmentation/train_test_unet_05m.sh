#!/bin/bash 
echo "hillshade"
mkdir /workspace/data/logfiles/UNet/05m/hillshade1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_05m/training/hillshade/ /workspace/data/final_data_05m/training/labels/ /workspace/data/logfiles/UNet/05m/hillshade1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_05m/testing/hillshade/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/UNet/05m/hillshade1/trained.h5 UNet /workspace/data/logfiles/UNet/05m/hillshade1/test.csv --classes=0,1


echo "maxelevationdeviation"
mkdir /workspace/data/logfiles/UNet/05m/maxelevationdeviation1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_05m/training/maxelevationdeviation/ /workspace/data/final_data_05m/training/labels/ /workspace/data/logfiles/UNet/05m/maxelevationdeviation1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_05m/testing/maxelevationdeviation/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/UNet/05m/maxelevationdeviation1/trained.h5 UNet /workspace/data/logfiles/UNet/05m/maxelevationdeviation1/test.csv --classes=0,1


echo "multiscaleelevationpercentile"
mkdir /workspace/data/logfiles/UNet/05m/multiscaleelevationpercentile1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_05m/training/multiscaleelevationpercentile/ /workspace/data/final_data_05m/training/labels/ /workspace/data/logfiles/UNet/05m/multiscaleelevationpercentile1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_05m/testing/multiscaleelevationpercentile/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/UNet/05m/multiscaleelevationpercentile1/trained.h5 UNet /workspace/data/logfiles/UNet/05m/multiscaleelevationpercentile1/test.csv --classes=0,1

echo "minimal_curvature"
mkdir /workspace/data/logfiles/UNet/05m/minimal_curvature1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_05m/training/minimal_curvature/ /workspace/data/final_data_05m/training/labels/ /workspace/data/logfiles/UNet/05m/minimal_curvature1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_05m/testing/minimal_curvature/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/UNet/05m/minimal_curvature1/trained.h5 UNet /workspace/data/logfiles/UNet/05m/minimal_curvature1/test.csv --classes=0,1

echo "maximal_curvature"
mkdir /workspace/data/logfiles/UNet/05m/maximal_curvature1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_05m/training/maximal_curvature/ /workspace/data/final_data_05m/training/labels/ /workspace/data/logfiles/UNet/05m/maximal_curvature1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_05m/testing/maximal_curvature/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/UNet/05m/maximal_curvature1/trained.h5 UNet /workspace/data/logfiles/UNet/05m/maximal_curvature1/test.csv --classes=0,1

echo "profile_curvature"
mkdir /workspace/data/logfiles/UNet/05m/profile_curvature1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_05m/training/profile_curvature/ /workspace/data/final_data_05m/training/labels/ /workspace/data/logfiles/UNet/05m/profile_curvature1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_05m/testing/profile_curvature/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/UNet/05m/profile_curvature1/trained.h5 UNet /workspace/data/logfiles/UNet/05m/profile_curvature1/test.csv --classes=0,1

echo "stdon"
mkdir /workspace/data/logfiles/UNet/05m/stdon1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_05m/training/stdon/ /workspace/data/final_data_05m/training/labels/ /workspace/data/logfiles/UNet/05m/stdon1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_05m/testing/stdon/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/UNet/05m/stdon1/trained.h5 UNet /workspace/data/logfiles/UNet/05m/stdon1/test.csv --classes=0,1

echo "multiscale_stdon"
mkdir /workspace/data/logfiles/UNet/05m/multiscale_stdon1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_05m/training/multiscale_stdon/ /workspace/data/final_data_05m/training/labels/ /workspace/data/logfiles/UNet/05m/multiscale_stdon1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_05m/testing/multiscale_stdon/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/UNet/05m/multiscale_stdon1/trained.h5 UNet /workspace/data/logfiles/UNet/05m/multiscale_stdon1/test.csv --classes=0,1

echo "elevation_above_pit"
mkdir /workspace/data/logfiles/UNet/05m/elevation_above_pit1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_05m/training/elevation_above_pit/ /workspace/data/final_data_05m/training/labels/ /workspace/data/logfiles/UNet/05m/elevation_above_pit1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_05m/testing/elevation_above_pit/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/UNet/05m/elevation_above_pit1/trained.h5 UNet /workspace/data/logfiles/UNet/05m/elevation_above_pit1/test.csv --classes=0,1

echo "depthinsink"
mkdir /workspace/data/logfiles/UNet/05m/depthinsink1/
python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_05m/training/depthinsink/ /workspace/data/final_data_05m/training/labels/ /workspace/data/logfiles/UNet/05m/depthinsink1/ --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_05m/testing/depthinsink/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/UNet/05m/depthinsink1/trained.h5 UNet /workspace/data/logfiles/UNet/05m/depthinsink1/test.csv --classes=0,1

#mkdir /workspace/data/logfiles/UNet/05m/combined1/
#python /workspace/code/semantic_segmentation/train.py -I /workspace/data/final_data_05m/training/hillshade/ -I /workspace/data/final_data_05m/training/maxelevationdeviation/ -I /workspace/data/final_data_05m/training/multiscaleelevationpercentile/ -I /workspace/data/final_data_05m/training/minimal_curvature/ -I /workspace/data/final_data_05m/training/maximal_curvature/ -I /workspace/data/final_data_05m/training/profile_curvature/ -I /workspace/data/final_data_05m/training/stdon/ -I /workspace/data/final_data_05m/training/multiscale_stdon/ -I /workspace/data/final_data_05m/training/elevation_above_pit/ -I /workspace/data/final_data_05m/training/depthinsink/ /workspace/data/final_data_05m/training/labels/ /workspace/data/logfiles/UNet/05m/combined1 --weighting="focal" UNet --seed=1 --epochs=50 --batch_size=32  --classes=0,1 --model_path='/workspace/data/logfiles/moon/unet/trained.h5'
#python /workspace/code/semantic_segmentation/evaluate.py -I /workspace/data/final_data_05m/testing/hillshade/ -I /workspace/data/final_data_05m/testing/maxelevationdeviation/ -I /workspace/data/final_data_05m/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data_05m/testing/minimal_curvature/ -I /workspace/data/final_data_05m/testing/maximal_curvature/ -I /workspace/data/final_data_05m/testing/profile_curvature/ -I /workspace/data/final_data_05m/testing/stdon/ -I /workspace/data/final_data_05m/testing/multiscale_stdon/ -I /workspace/data/final_data_05m/testing/elevation_above_pit/ -I /workspace/data/final_data_05m/testing/depthinsink/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/logfiles/UNet/05m/combined1/trained.h5 UNet /workspace/data/logfiles/UNet/05m/combined1/test.csv --classes=0,1



