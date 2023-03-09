#!/bin/bash 
echo "hillshade"
mkdir /workspace/data/logfiles/1m/hillshade1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/1m/hillshade1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/1m/hillshade1/trained.h5 /workspace/data/logfiles/1m/hillshade1/test.csv --classes=0,1

echo "Maximum elevation deviation"
mkdir /workspace/data/logfiles/1m/maxelevationdeviation1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maxelevationdeviation/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/1m/maxelevationdeviation1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/1m/maxelevationdeviation1/trained.h5 /workspace/data/logfiles/1m/maxelevationdeviation1/test.csv --classes=0,1

echo "Multiscale elevation percentile"
mkdir /workspace/data/logfiles/1m/Multiscale_elevation_percentile1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/Multiscale_elevation_percentile/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/1m/Multiscale_elevation_percentile1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/Multiscale_elevation_percentile/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/1m/Multiscale_elevation_percentile1/trained.h5 /workspace/data/logfiles/1m/Multiscale_elevation_percentile1/test.csv --classes=0,1

echo "Minimal curvature"
mkdir /workspace/data/logfiles/1m/Minimal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/Minimal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/1m/Minimal_curvature1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/Minimal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/1m/Minimal_curvature1/trained.h5 /workspace/data/logfiles/1m/Minimal_curvature1/test.csv --classes=0,1

echo "Maximal curvature"
mkdir /workspace/data/logfiles/1m/Maximal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/Maximal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/1m/Maximal_curvature1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/Maximal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/1m/Maximal_curvature1/trained.h5 /workspace/data/logfiles/1m/Maximal_curvature1/test.csv --classes=0,1

echo "Profile curvature"
mkdir /workspace/data/logfiles/1m/Profile_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/Profile_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/1m/Profile_curvature1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/Profile_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/1m/Profile_curvature1/trained.h5 /workspace/data/logfiles/1m/Profile_curvature1/test.csv --classes=0,1

echo "Spherical standard deviation of normal"
mkdir /workspace/data/logfiles/1m/Spherical_standard_deviation_of_normal1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/Spherical_standard_deviation_of_normal/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/1m/Spherical_standard_deviation_of_normal1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/Spherical_standard_deviation_of_normal/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/1m/Spherical_standard_deviation_of_normal1/trained.h5 /workspace/data/logfiles/1m/Spherical_standard_deviation_of_normal1/test.csv --classes=0,1

echo "multiscale_stdon"
mkdir /workspace/data/logfiles/1m/multiscale_stdon1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscale_stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/1m/multiscale_stdon1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/1m/multiscale_stdon1/trained.h5 /workspace/data/logfiles/1m/multiscale_stdon1/test.csv --classes=0,1

echo "Elevation above pit"
mkdir /workspace/data/logfiles/1m/Elevation_above_pit1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/Elevation_above_pit/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/1m/Elevation_above_pit1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/Elevation_above_pit/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/1m/Elevation_above_pit1/trained.h5 /workspace/data/logfiles/1m/Elevation_above_pit1/test.csv --classes=0,1

echo "depthinsink"
mkdir /workspace/data/logfiles/1m/depthinsink1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/1m/depthinsink1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/1m/depthinsink1/trained.h5 /workspace/data/logfiles/1m/depthinsink1/test.csv --classes=0,1

echo "everything combined"
mkdir /workspace/data/logfiles/1m/combined1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ -I /workspace/data/final_data_1m/training/maxelevationdeviation/ -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/training/minimal_curvature/ -I /workspace/data/final_data_1m/training/maximal_curvature/ -I /workspace/data/final_data_1m/training/profile_curvature/ -I /workspace/data/final_data_1m/training/stdon/ -I /workspace/data/final_data_1m/training/multiscale_stdon/ -I /workspace/data/final_data_1m/training/elevation_above_pit/ -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/data/logfiles/1m/combined1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/testing/minimal_curvature/ -I /workspace/data/final_data_1m/testing/maximal_curvature/ -I /workspace/data/final_data_1m/testing/profile_curvature/ -I /workspace/data/final_data_1m/testing/stdon/ -I /workspace/data/final_data_1m/testing/multiscale_stdon/ -I /workspace/data/final_data_1m/testing/elevation_above_pit/ -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/data/logfiles/1m/combined1/trained.h5 /workspace/data/logfiles/1m/combined1/test.csv --classes=0,1


