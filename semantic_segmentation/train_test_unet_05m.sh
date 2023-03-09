#!/bin/bash 
echo "hillshade"
mkdir /workspace/data/logfiles/05m/hillshade1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/hillshade/ /workspace/data/final_data/training/labels/ /workspace/data/logfiles/05m/hillshade1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/hillshade/ /workspace/data/final_data/testing/labels/ /workspace/data/logfiles/05m/hillshade1/trained.h5 /workspace/data/logfiles/05m/hillshade1/test.csv --classes=0,1

echo "Maximum elevation deviation"
mkdir /workspace/data/logfiles/05m/maxelevationdeviation1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/maxelevationdeviation/ /workspace/data/final_data/training/labels/ /workspace/data/logfiles/05m/maxelevationdeviation1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/maxelevationdeviation/ /workspace/data/final_data/testing/labels/ /workspace/data/logfiles/05m/maxelevationdeviation1/trained.h5 /workspace/data/logfiles/05m/maxelevationdeviation1/test.csv --classes=0,1

echo "Multiscale elevation percentile"
mkdir /workspace/data/logfiles/05m/Multiscale_elevation_percentile1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/Multiscale_elevation_percentile/ /workspace/data/final_data/training/labels/ /workspace/data/logfiles/05m/Multiscale_elevation_percentile1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/Multiscale_elevation_percentile/ /workspace/data/final_data/testing/labels/ /workspace/data/logfiles/05m/Multiscale_elevation_percentile1/trained.h5 /workspace/data/logfiles/05m/Multiscale_elevation_percentile1/test.csv --classes=0,1

echo "Minimal curvature"
mkdir /workspace/data/logfiles/05m/Minimal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/Minimal_curvature/ /workspace/data/final_data/training/labels/ /workspace/data/logfiles/05m/Minimal_curvature1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/Minimal_curvature/ /workspace/data/final_data/testing/labels/ /workspace/data/logfiles/05m/Minimal_curvature1/trained.h5 /workspace/data/logfiles/05m/Minimal_curvature1/test.csv --classes=0,1

echo "Maximal curvature"
mkdir /workspace/data/logfiles/05m/Maximal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/Maximal_curvature/ /workspace/data/final_data/training/labels/ /workspace/data/logfiles/05m/Maximal_curvature1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/Maximal_curvature/ /workspace/data/final_data/testing/labels/ /workspace/data/logfiles/05m/Maximal_curvature1/trained.h5 /workspace/data/logfiles/05m/Maximal_curvature1/test.csv --classes=0,1

echo "Profile curvature"
mkdir /workspace/data/logfiles/05m/Profile_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/Profile_curvature/ /workspace/data/final_data/training/labels/ /workspace/data/logfiles/05m/Profile_curvature1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/Profile_curvature/ /workspace/data/final_data/testing/labels/ /workspace/data/logfiles/05m/Profile_curvature1/trained.h5 /workspace/data/logfiles/05m/Profile_curvature1/test.csv --classes=0,1

echo "Spherical standard deviation of normal"
mkdir /workspace/data/logfiles/05m/Spherical_standard_deviation_of_normal1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/Spherical_standard_deviation_of_normal/ /workspace/data/final_data/training/labels/ /workspace/data/logfiles/05m/Spherical_standard_deviation_of_normal1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/Spherical_standard_deviation_of_normal/ /workspace/data/final_data/testing/labels/ /workspace/data/logfiles/05m/Spherical_standard_deviation_of_normal1/trained.h5 /workspace/data/logfiles/05m/Spherical_standard_deviation_of_normal1/test.csv --classes=0,1

echo "multiscale_stdon"
mkdir /workspace/data/logfiles/05m/multiscale_stdon1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/multiscale_stdon/ /workspace/data/final_data/training/labels/ /workspace/data/logfiles/05m/multiscale_stdon1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/multiscale_stdon/ /workspace/data/final_data/testing/labels/ /workspace/data/logfiles/05m/multiscale_stdon1/trained.h5 /workspace/data/logfiles/05m/multiscale_stdon1/test.csv --classes=0,1

echo "Elevation above pit"
mkdir /workspace/data/logfiles/05m/Elevation_above_pit1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/Elevation_above_pit/ /workspace/data/final_data/training/labels/ /workspace/data/logfiles/05m/Elevation_above_pit1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/Elevation_above_pit/ /workspace/data/final_data/testing/labels/ /workspace/data/logfiles/05m/Elevation_above_pit1/trained.h5 /workspace/data/logfiles/05m/Elevation_above_pit1/test.csv --classes=0,1

echo "depthinsink"
mkdir /workspace/data/logfiles/05m/depthinsink1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/depthinsink/ /workspace/data/final_data/training/labels/ /workspace/data/logfiles/05m/depthinsink1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/depthinsink/ /workspace/data/final_data/testing/labels/ /workspace/data/logfiles/05m/depthinsink1/trained.h5 /workspace/data/logfiles/05m/depthinsink1/test.csv --classes=0,1

echo "everything combined"
mkdir /workspace/data/logfiles/05m/combined1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/hillshade/ -I /workspace/data/final_data/training/maxelevationdeviation/ -I /workspace/data/final_data/training/multiscaleelevationpercentile/ -I /workspace/data/final_data/training/minimal_curvature/ -I /workspace/data/final_data/training/maximal_curvature/ -I /workspace/data/final_data/training/profile_curvature/ -I /workspace/data/final_data/training/stdon/ -I /workspace/data/final_data/training/multiscale_stdon/ -I /workspace/data/final_data/training/elevation_above_pit/ -I /workspace/data/final_data/training/depthinsink/ /workspace/data/final_data/training/labels/ /workspace/data/logfiles/05m/combined1/ --weighting="mfb" --seed=42 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/hillshade/ -I /workspace/data/final_data/testing/maxelevationdeviation/ -I /workspace/data/final_data/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data/testing/minimal_curvature/ -I /workspace/data/final_data/testing/maximal_curvature/ -I /workspace/data/final_data/testing/profile_curvature/ -I /workspace/data/final_data/testing/stdon/ -I /workspace/data/final_data/testing/multiscale_stdon/ -I /workspace/data/final_data/testing/elevation_above_pit/ -I /workspace/data/final_data/testing/depthinsink/ /workspace/data/final_data/testing/labels/ /workspace/data/logfiles/05m/combined1/trained.h5 /workspace/data/logfiles/05m/combined1/test.csv --classes=0,1


