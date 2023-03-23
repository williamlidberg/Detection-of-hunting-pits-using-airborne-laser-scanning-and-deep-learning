#!/bin/bash 
# SEED 0 depth3
echo "hillshade"
mkdir /workspace/logfiles/seed0_depth3/1m/hillshade1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed0_depth3/1m/hillshade1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed0_depth3/1m/hillshade1/trained.h5 /workspace/logfiles/seed0_depth3/1m/hillshade1/test.csv --classes=0,1 --depth=3

echo "maximum elevation deviation"
mkdir /workspace/logfiles/seed0_depth3/1m/maxelevationdeviation1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maxelevationdeviation/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed0_depth3/1m/maxelevationdeviation1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed0_depth3/1m/maxelevationdeviation1/trained.h5 /workspace/logfiles/seed0_depth3/1m/maxelevationdeviation1/test.csv --classes=0,1 --depth=3

echo "multiscale elevation percentile"
mkdir /workspace/logfiles/seed0_depth3/1m/multiscale_elevation_percentile1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed0_depth3/1m/multiscale_elevation_percentile1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed0_depth3/1m/multiscale_elevation_percentile1/trained.h5 /workspace/logfiles/seed0_depth3/1m/multiscale_elevation_percentile1/test.csv --classes=0,1 --depth=3

echo "minimal curvature"
mkdir /workspace/logfiles/seed0_depth3/1m/minimal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/minimal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed0_depth3/1m/minimal_curvature1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/minimal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed0_depth3/1m/minimal_curvature1/trained.h5 /workspace/logfiles/seed0_depth3/1m/minimal_curvature1/test.csv --classes=0,1 --depth=3

echo "maximal curvature"
mkdir /workspace/logfiles/seed0_depth3/1m/maximal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maximal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed0_depth3/1m/maximal_curvature1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maximal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed0_depth3/1m/maximal_curvature1/trained.h5 /workspace/logfiles/seed0_depth3/1m/maximal_curvature1/test.csv --classes=0,1 --depth=3

echo "profile curvature"
mkdir /workspace/logfiles/seed0_depth3/1m/profile_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/profile_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed0_depth3/1m/profile_curvature1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/profile_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed0_depth3/1m/profile_curvature1/trained.h5 /workspace/logfiles/seed0_depth3/1m/profile_curvature1/test.csv --classes=0,1 --depth=3

echo "spherical standard deviation of normal"
mkdir /workspace/logfiles/seed0_depth3/1m/spherical_standard_deviation_of_normal1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed0_depth3/1m/spherical_standard_deviation_of_normal1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed0_depth3/1m/spherical_standard_deviation_of_normal1/trained.h5 /workspace/logfiles/seed0_depth3/1m/spherical_standard_deviation_of_normal1/test.csv --classes=0,1 --depth=3

echo "multiscale_stdon"
mkdir /workspace/logfiles/seed0_depth3/1m/multiscale_stdon1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscale_stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed0_depth3/1m/multiscale_stdon1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed0_depth3/1m/multiscale_stdon1/trained.h5 /workspace/logfiles/seed0_depth3/1m/multiscale_stdon1/test.csv --classes=0,1 --depth=3

echo "elevation above pit"
mkdir /workspace/logfiles/seed0_depth3/1m/elevation_above_pit1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/elevation_above_pit/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed0_depth3/1m/elevation_above_pit1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/elevation_above_pit/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed0_depth3/1m/elevation_above_pit1/trained.h5 /workspace/logfiles/seed0_depth3/1m/elevation_above_pit1/test.csv --classes=0,1 --depth=3

echo "depthinsink"
mkdir /workspace/logfiles/seed0_depth3/1m/depthinsink1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed0_depth3/1m/depthinsink1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed0_depth3/1m/depthinsink1/trained.h5 /workspace/logfiles/seed0_depth3/1m/depthinsink1/test.csv --classes=0,1 --depth=3

echo "everything combined"
mkdir /workspace/logfiles/seed0_depth3/1m/combined1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ -I /workspace/data/final_data_1m/training/maxelevationdeviation/ -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/training/minimal_curvature/ -I /workspace/data/final_data_1m/training/maximal_curvature/ -I /workspace/data/final_data_1m/training/profile_curvature/ -I /workspace/data/final_data_1m/training/stdon/ -I /workspace/data/final_data_1m/training/multiscale_stdon/ -I /workspace/data/final_data_1m/training/elevation_above_pit/ -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed0_depth3/1m/combined1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/testing/minimal_curvature/ -I /workspace/data/final_data_1m/testing/maximal_curvature/ -I /workspace/data/final_data_1m/testing/profile_curvature/ -I /workspace/data/final_data_1m/testing/stdon/ -I /workspace/data/final_data_1m/testing/multiscale_stdon/ -I /workspace/data/final_data_1m/testing/elevation_above_pit/ -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed0_depth3/1m/combined1/trained.h5 /workspace/logfiles/seed0_depth3/1m/combined1/test.csv --classes=0,1 --depth=3


# SEED 1 depth3
echo "hillshade"
mkdir /workspace/logfiles/seed1_depth3/1m/hillshade1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed1_depth3/1m/hillshade1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed1_depth3/1m/hillshade1/trained.h5 /workspace/logfiles/seed1_depth3/1m/hillshade1/test.csv --classes=0,1 --depth=3

echo "maximum elevation deviation"
mkdir /workspace/logfiles/seed1_depth3/1m/maxelevationdeviation1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maxelevationdeviation/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed1_depth3/1m/maxelevationdeviation1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed1_depth3/1m/maxelevationdeviation1/trained.h5 /workspace/logfiles/seed1_depth3/1m/maxelevationdeviation1/test.csv --classes=0,1 --depth=3

echo "multiscale elevation percentile"
mkdir /workspace/logfiles/seed1_depth3/1m/multiscale_elevation_percentile1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed1_depth3/1m/multiscale_elevation_percentile1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed1_depth3/1m/multiscale_elevation_percentile1/trained.h5 /workspace/logfiles/seed1_depth3/1m/multiscale_elevation_percentile1/test.csv --classes=0,1 --depth=3

echo "minimal curvature"
mkdir /workspace/logfiles/seed1_depth3/1m/minimal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/minimal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed1_depth3/1m/minimal_curvature1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/minimal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed1_depth3/1m/minimal_curvature1/trained.h5 /workspace/logfiles/seed1_depth3/1m/minimal_curvature1/test.csv --classes=0,1 --depth=3

echo "maximal curvature"
mkdir /workspace/logfiles/seed1_depth3/1m/maximal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maximal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed1_depth3/1m/maximal_curvature1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maximal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed1_depth3/1m/maximal_curvature1/trained.h5 /workspace/logfiles/seed1_depth3/1m/maximal_curvature1/test.csv --classes=0,1 --depth=3

echo "profile curvature"
mkdir /workspace/logfiles/seed1_depth3/1m/profile_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/profile_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed1_depth3/1m/profile_curvature1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/profile_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed1_depth3/1m/profile_curvature1/trained.h5 /workspace/logfiles/seed1_depth3/1m/profile_curvature1/test.csv --classes=0,1 --depth=3

echo "spherical standard deviation of normal"
mkdir /workspace/logfiles/seed1_depth3/1m/spherical_standard_deviation_of_normal1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed1_depth3/1m/spherical_standard_deviation_of_normal1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed1_depth3/1m/spherical_standard_deviation_of_normal1/trained.h5 /workspace/logfiles/seed1_depth3/1m/spherical_standard_deviation_of_normal1/test.csv --classes=0,1 --depth=3

echo "multiscale_stdon"
mkdir /workspace/logfiles/seed1_depth3/1m/multiscale_stdon1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscale_stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed1_depth3/1m/multiscale_stdon1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed1_depth3/1m/multiscale_stdon1/trained.h5 /workspace/logfiles/seed1_depth3/1m/multiscale_stdon1/test.csv --classes=0,1 --depth=3

echo "elevation above pit"
mkdir /workspace/logfiles/seed1_depth3/1m/elevation_above_pit1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/elevation_above_pit/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed1_depth3/1m/elevation_above_pit1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/elevation_above_pit/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed1_depth3/1m/elevation_above_pit1/trained.h5 /workspace/logfiles/seed1_depth3/1m/elevation_above_pit1/test.csv --classes=0,1 --depth=3

echo "depthinsink"
mkdir /workspace/logfiles/seed1_depth3/1m/depthinsink1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed1_depth3/1m/depthinsink1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed1_depth3/1m/depthinsink1/trained.h5 /workspace/logfiles/seed1_depth3/1m/depthinsink1/test.csv --classes=0,1 --depth=3

echo "everything combined"
mkdir /workspace/logfiles/seed1_depth3/1m/combined1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ -I /workspace/data/final_data_1m/training/maxelevationdeviation/ -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/training/minimal_curvature/ -I /workspace/data/final_data_1m/training/maximal_curvature/ -I /workspace/data/final_data_1m/training/profile_curvature/ -I /workspace/data/final_data_1m/training/stdon/ -I /workspace/data/final_data_1m/training/multiscale_stdon/ -I /workspace/data/final_data_1m/training/elevation_above_pit/ -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed1_depth3/1m/combined1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/testing/minimal_curvature/ -I /workspace/data/final_data_1m/testing/maximal_curvature/ -I /workspace/data/final_data_1m/testing/profile_curvature/ -I /workspace/data/final_data_1m/testing/stdon/ -I /workspace/data/final_data_1m/testing/multiscale_stdon/ -I /workspace/data/final_data_1m/testing/elevation_above_pit/ -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed1_depth3/1m/combined1/trained.h5 /workspace/logfiles/seed1_depth3/1m/combined1/test.csv --classes=0,1 --depth=3

# seed2 depth3
echo "hillshade"
mkdir /workspace/logfiles/seed2_depth3/1m/hillshade1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed2_depth3/1m/hillshade1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed2_depth3/1m/hillshade1/trained.h5 /workspace/logfiles/seed2_depth3/1m/hillshade1/test.csv --classes=0,1 --depth=3

echo "maximum elevation deviation"
mkdir /workspace/logfiles/seed2_depth3/1m/maxelevationdeviation1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maxelevationdeviation/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed2_depth3/1m/maxelevationdeviation1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed2_depth3/1m/maxelevationdeviation1/trained.h5 /workspace/logfiles/seed2_depth3/1m/maxelevationdeviation1/test.csv --classes=0,1 --depth=3

echo "multiscale elevation percentile"
mkdir /workspace/logfiles/seed2_depth3/1m/multiscale_elevation_percentile1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed2_depth3/1m/multiscale_elevation_percentile1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed2_depth3/1m/multiscale_elevation_percentile1/trained.h5 /workspace/logfiles/seed2_depth3/1m/multiscale_elevation_percentile1/test.csv --classes=0,1 --depth=3

echo "minimal curvature"
mkdir /workspace/logfiles/seed2_depth3/1m/minimal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/minimal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed2_depth3/1m/minimal_curvature1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/minimal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed2_depth3/1m/minimal_curvature1/trained.h5 /workspace/logfiles/seed2_depth3/1m/minimal_curvature1/test.csv --classes=0,1 --depth=3

echo "maximal curvature"
mkdir /workspace/logfiles/seed2_depth3/1m/maximal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maximal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed2_depth3/1m/maximal_curvature1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maximal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed2_depth3/1m/maximal_curvature1/trained.h5 /workspace/logfiles/seed2_depth3/1m/maximal_curvature1/test.csv --classes=0,1 --depth=3

echo "profile curvature"
mkdir /workspace/logfiles/seed2_depth3/1m/profile_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/profile_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed2_depth3/1m/profile_curvature1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/profile_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed2_depth3/1m/profile_curvature1/trained.h5 /workspace/logfiles/seed2_depth3/1m/profile_curvature1/test.csv --classes=0,1 --depth=3

echo "spherical standard deviation of normal"
mkdir /workspace/logfiles/seed2_depth3/1m/spherical_standard_deviation_of_normal1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed2_depth3/1m/spherical_standard_deviation_of_normal1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed2_depth3/1m/spherical_standard_deviation_of_normal1/trained.h5 /workspace/logfiles/seed2_depth3/1m/spherical_standard_deviation_of_normal1/test.csv --classes=0,1 --depth=3

echo "multiscale_stdon"
mkdir /workspace/logfiles/seed2_depth3/1m/multiscale_stdon1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscale_stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed2_depth3/1m/multiscale_stdon1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed2_depth3/1m/multiscale_stdon1/trained.h5 /workspace/logfiles/seed2_depth3/1m/multiscale_stdon1/test.csv --classes=0,1 --depth=3

echo "elevation above pit"
mkdir /workspace/logfiles/seed2_depth3/1m/elevation_above_pit1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/elevation_above_pit/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed2_depth3/1m/elevation_above_pit1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/elevation_above_pit/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed2_depth3/1m/elevation_above_pit1/trained.h5 /workspace/logfiles/seed2_depth3/1m/elevation_above_pit1/test.csv --classes=0,1 --depth=3

echo "depthinsink"
mkdir /workspace/logfiles/seed2_depth3/1m/depthinsink1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed2_depth3/1m/depthinsink1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed2_depth3/1m/depthinsink1/trained.h5 /workspace/logfiles/seed2_depth3/1m/depthinsink1/test.csv --classes=0,1 --depth=3

echo "everything combined"
mkdir /workspace/logfiles/seed2_depth3/1m/combined1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ -I /workspace/data/final_data_1m/training/maxelevationdeviation/ -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/training/minimal_curvature/ -I /workspace/data/final_data_1m/training/maximal_curvature/ -I /workspace/data/final_data_1m/training/profile_curvature/ -I /workspace/data/final_data_1m/training/stdon/ -I /workspace/data/final_data_1m/training/multiscale_stdon/ -I /workspace/data/final_data_1m/training/elevation_above_pit/ -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/seed2_depth3/1m/combined1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/testing/minimal_curvature/ -I /workspace/data/final_data_1m/testing/maximal_curvature/ -I /workspace/data/final_data_1m/testing/profile_curvature/ -I /workspace/data/final_data_1m/testing/stdon/ -I /workspace/data/final_data_1m/testing/multiscale_stdon/ -I /workspace/data/final_data_1m/testing/elevation_above_pit/ -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/seed2_depth3/1m/combined1/trained.h5 /workspace/logfiles/seed2_depth3/1m/combined1/test.csv --classes=0,1 --depth=3
