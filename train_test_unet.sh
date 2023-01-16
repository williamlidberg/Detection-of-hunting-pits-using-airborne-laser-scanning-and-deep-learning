#!/bin/bash 
echo "hillshade"
mkdir /workspace/data/logfiles/pits/hillshade1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/hillshade/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/hillshade1/ --weighting="0.01,1" --seed=40 --epochs 100 --batch_size=4
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/hillshade/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/hillshade1/trained.h5 /workspace/data/logfiles/pits/hillshade1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/hillshade1/valid_imgs.txt --classes=0,1

echo "Maximum elevation deviation"
mkdir /workspace/data/logfiles/pits/Maximum_elevation_deviation1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/maxelevationdeviation/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Maximum_elevation_deviation1/ --weighting="0.01,1" --seed=40 --epochs 100 --batch_size=4
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/maxelevationdeviation/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Maximum_elevation_deviation1/trained.h5 /workspace/data/logfiles/pits/Maximum_elevation_deviation1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/Maximum_elevation_deviation1/valid_imgs.txt --classes=0,1

echo "Multiscale elevation percentile"
mkdir /workspace/data/logfiles/pits/Multiscale_elevation_percentile1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/multiscaleelevationpercentile/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Multiscale_elevation_percentile1/ --weighting="0.01,1" --seed=40 --epochs 100 --batch_size=4
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/multiscaleelevationpercentile/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Multiscale_elevation_percentile1/trained.h5 /workspace/data/logfiles/pits/Multiscale_elevation_percentile1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/Multiscale_elevation_percentile1/valid_imgs.txt --classes=0,1

echo "Minimal curvature"
mkdir /workspace/data/logfiles/pits/Minimal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/minimal_curvature/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Minimal_curvature1/ --weighting="0.01,1" --seed=40 --epochs 100 --batch_size=4
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/minimal_curvature/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Minimal_curvature1/trained.h5 /workspace/data/logfiles/pits/Minimal_curvature1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/Minimal_curvature1/valid_imgs.txt --classes=0,1

echo "Maximal curvature"
mkdir /workspace/data/logfiles/pits/Maximal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/maximal_curvature/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Maximal_curvature1/ --weighting="0.01,1" --seed=40 --epochs 100 --batch_size=4
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/maximal_curvature/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Maximal_curvature1/trained.h5 /workspace/data/logfiles/pits/Maximal_curvature1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/Maximal_curvature1/valid_imgs.txt --classes=0,1

echo "Profile curvature"
mkdir /workspace/data/logfiles/pits/Profile_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/profile_curvature/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Profile_curvature1/ --weighting="0.01,1" --seed=40 --epochs 100 --batch_size=4
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/profile_curvature/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Profile_curvature1/trained.h5 /workspace/data/logfiles/pits/Profile_curvature1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/Profile_curvature1/valid_imgs.txt --classes=0,1

echo "Spherical standard deviation of normal"
mkdir /workspace/data/logfiles/pits/Spherical_standard_deviation_of_normal1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/stdon/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Spherical_standard_deviation_of_normal1/ --weighting="0.01,1" --seed=40 --epochs 100 --batch_size=4
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/stdon/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Spherical_standard_deviation_of_normal1/trained.h5 /workspace/data/logfiles/pits/Spherical_standard_deviation_of_normal1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/Spherical_standard_deviation_of_normal1/valid_imgs.txt --classes=0,1

echo "Multiscale standard deviation of normal"
mkdir /workspace/data/logfiles/pits/Multiscale_standard_deviation_of_normal1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/multiscale_stdon/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Multiscale_standard_deviation_of_normal1/ --weighting="0.01,1" --seed=40 --epochs 100 --batch_size=4
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/multiscale_stdon/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Multiscale_standard_deviation_of_normal1/trained.h5 /workspace/data/logfiles/pits/Multiscale_standard_deviation_of_normal1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/Multiscale_standard_deviation_of_normal1/valid_imgs.txt --classes=0,1

echo "Elevation above pit"
mkdir /workspace/data/logfiles/pits/Elevation_above_pit1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/elevation_above_pit/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Elevation_above_pit1/ --weighting="0.01,1" --seed=40 --epochs 100 --batch_size=4
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/elevation_above_pit/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/Elevation_above_pit1/trained.h5 /workspace/data/logfiles/pits/Elevation_above_pit1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/Elevation_above_pit1/valid_imgs.txt --classes=0,1

echo "depthinsink"
mkdir /workspace/data/logfiles/pits/depthinsink1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/depthinsink/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/depthinsink1/ --weighting="0.01,1" --seed=40 --epochs 100 --batch_size=4
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/depthinsink/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/depthinsink1/trained.h5 /workspace/data/logfiles/pits/depthinsink1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/depthinsink1/valid_imgs.txt --classes=0,1

echo "everything combined"
mkdir /workspace/data/logfiles/pits/combined1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/hillshade/ -I /workspace/data/split_data_pits/maxelevationdeviation/ -I /workspace/data/split_data_pits/multiscaleelevationpercentile/ -I /workspace/data/split_data_pits/minimal_curvature/ -I /workspace/data/split_data_pits/maximal_curvature/ -I /workspace/data/split_data_pits/profile_curvature/ -I /workspace/data/split_data_pits/stdon/ -I /workspace/data/split_data_pits/multiscale_stdon/ -I /workspace/data/split_data_pits/elevation_above_pit/ -I /workspace/data/split_data_pits/depthinsink/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/combined1/ --weighting="0.01,1" --seed=40 --epochs 100
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/hillshade/ -I /workspace/data/split_data_pits/maxelevationdeviation/ -I /workspace/data/split_data_pits/multiscaleelevationpercentile/ -I /workspace/data/split_data_pits/minimal_curvature/ -I /workspace/data/split_data_pits/maximal_curvature/ -I /workspace/data/split_data_pits/profile_curvature/ -I /workspace/data/split_data_pits/stdon/ -I /workspace/data/split_data_pits/multiscale_stdon/ -I /workspace/data/split_data_pits/elevation_above_pit/ -I /workspace/data/split_data_pits/depthinsink/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/combined1/trained.h5 /workspace/data/logfiles/pits/combined1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/combined1/valid_imgs.txt --classes=0,1

echo "combined above 0.4"
mkdir /workspace/data/logfiles/pits/combined1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/hillshade/ -I /workspace/data/split_data_pits/maxelevationdeviation/ -I /workspace/data/split_data_pits/multiscaleelevationpercentile/ -I /workspace/data/split_data_pits/minimal_curvature/ -I /workspace/data/split_data_pits/depthinsink/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/combined1/ --weighting="0.01,1" --seed=40 --epochs 100
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/hillshade/ -I /workspace/data/split_data_pits/maxelevationdeviation/ -I /workspace/data/split_data_pits/multiscaleelevationpercentile/ -I /workspace/data/split_data_pits/minimal_curvature/ -I /workspace/data/split_data_pits/depthinsink/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/combined1/trained.h5 /workspace/data/logfiles/pits/combined1/eval.csv --selected_imgs=/workspace/data/logfiles/pits/combined1/valid_imgs.txt --classes=0,1

echo "everything combined"
mkdir /workspace/data/logfiles/pits/combined2/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/split_data_pits/hillshade/ -I /workspace/data/split_data_pits/maxelevationdeviation/ -I /workspace/data/split_data_pits/multiscaleelevationpercentile/ -I /workspace/data/split_data_pits/minimal_curvature/ -I /workspace/data/split_data_pits/maximal_curvature/ -I /workspace/data/split_data_pits/profile_curvature/ -I /workspace/data/split_data_pits/multiscale_stdon/ -I /workspace/data/split_data_pits/depthinsink/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/combined2/ --weighting="0.01,1" --seed=40 --epochs 100
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/split_data_pits/hillshade/ -I /workspace/data/split_data_pits/maxelevationdeviation/ -I /workspace/data/split_data_pits/multiscaleelevationpercentile/ -I /workspace/data/split_data_pits/minimal_curvature/ -I /workspace/data/split_data_pits/maximal_curvature/ -I /workspace/data/split_data_pits/profile_curvature/ -I /workspace/data/split_data_pits/multiscale_stdon/ -I /workspace/data/split_data_pits/depthinsink/ /workspace/data/split_data_pits/labels/ /workspace/data/logfiles/pits/combined2/trained.h5 /workspace/data/logfiles/pits/combined2/eval.csv --selected_imgs=/workspace/data/logfiles/pits/combined1/valid_imgs.txt --classes=0,1
