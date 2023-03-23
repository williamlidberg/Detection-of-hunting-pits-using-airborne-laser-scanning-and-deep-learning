#!/bin/bash 
# SEED 0 depth3
echo "hillshade"
mkdir /workspace/logfiles/seed0_depth3/05m/hillshade1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/hillshade/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed0_depth3/05m/hillshade1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/hillshade/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed0_depth3/05m/hillshade1/trained.h5 /workspace/logfiles/seed0_depth3/05m/hillshade1/test.csv --classes=0,1 --depth=3

echo "maximum elevation deviation"
mkdir /workspace/logfiles/seed0_depth3/05m/maxelevationdeviation1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/maxelevationdeviation/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed0_depth3/05m/maxelevationdeviation1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/maxelevationdeviation/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed0_depth3/05m/maxelevationdeviation1/trained.h5 /workspace/logfiles/seed0_depth3/05m/maxelevationdeviation1/test.csv --classes=0,1 --depth=3

echo "multiscale elevation percentile"
mkdir /workspace/logfiles/seed0_depth3/05m/multiscale_elevation_percentile1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/multiscaleelevationpercentile/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed0_depth3/05m/multiscale_elevation_percentile1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/multiscaleelevationpercentile/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed0_depth3/05m/multiscale_elevation_percentile1/trained.h5 /workspace/logfiles/seed0_depth3/05m/multiscale_elevation_percentile1/test.csv --classes=0,1 --depth=3

echo "minimal curvature"
mkdir /workspace/logfiles/seed0_depth3/05m/minimal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/minimal_curvature/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed0_depth3/05m/minimal_curvature1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/minimal_curvature/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed0_depth3/05m/minimal_curvature1/trained.h5 /workspace/logfiles/seed0_depth3/05m/minimal_curvature1/test.csv --classes=0,1 --depth=3

echo "maximal curvature"
mkdir /workspace/logfiles/seed0_depth3/05m/maximal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/maximal_curvature/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed0_depth3/05m/maximal_curvature1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/maximal_curvature/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed0_depth3/05m/maximal_curvature1/trained.h5 /workspace/logfiles/seed0_depth3/05m/maximal_curvature1/test.csv --classes=0,1 --depth=3

echo "profile curvature"
mkdir /workspace/logfiles/seed0_depth3/05m/profile_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/profile_curvature/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed0_depth3/05m/profile_curvature1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/profile_curvature/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed0_depth3/05m/profile_curvature1/trained.h5 /workspace/logfiles/seed0_depth3/05m/profile_curvature1/test.csv --classes=0,1 --depth=3

echo "spherical standard deviation of normal"
mkdir /workspace/logfiles/seed0_depth3/05m/spherical_standard_deviation_of_normal1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/stdon/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed0_depth3/05m/spherical_standard_deviation_of_normal1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/stdon/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed0_depth3/05m/spherical_standard_deviation_of_normal1/trained.h5 /workspace/logfiles/seed0_depth3/05m/spherical_standard_deviation_of_normal1/test.csv --classes=0,1 --depth=3

echo "multiscale_stdon"
mkdir /workspace/logfiles/seed0_depth3/05m/multiscale_stdon1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/multiscale_stdon/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed0_depth3/05m/multiscale_stdon1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/multiscale_stdon/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed0_depth3/05m/multiscale_stdon1/trained.h5 /workspace/logfiles/seed0_depth3/05m/multiscale_stdon1/test.csv --classes=0,1 --depth=3

echo "elevation above pit"
mkdir /workspace/logfiles/seed0_depth3/05m/elevation_above_pit1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/elevation_above_pit/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed0_depth3/05m/elevation_above_pit1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/elevation_above_pit/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed0_depth3/05m/elevation_above_pit1/trained.h5 /workspace/logfiles/seed0_depth3/05m/elevation_above_pit1/test.csv --classes=0,1 --depth=3

echo "depthinsink"
mkdir /workspace/logfiles/seed0_depth3/05m/depthinsink1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/depthinsink/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed0_depth3/05m/depthinsink1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/depthinsink/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed0_depth3/05m/depthinsink1/trained.h5 /workspace/logfiles/seed0_depth3/05m/depthinsink1/test.csv --classes=0,1 --depth=3

echo "everything combined"
mkdir /workspace/logfiles/seed0_depth3/05m/combined1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/hillshade/ -I /workspace/data/final_data/training/maxelevationdeviation/ -I /workspace/data/final_data/training/multiscaleelevationpercentile/ -I /workspace/data/final_data/training/minimal_curvature/ -I /workspace/data/final_data/training/maximal_curvature/ -I /workspace/data/final_data/training/profile_curvature/ -I /workspace/data/final_data/training/stdon/ -I /workspace/data/final_data/training/multiscale_stdon/ -I /workspace/data/final_data/training/elevation_above_pit/ -I /workspace/data/final_data/training/depthinsink/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed0_depth3/05m/combined1/ --weighting="mfb" --seed=0 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/hillshade/ -I /workspace/data/final_data/testing/maxelevationdeviation/ -I /workspace/data/final_data/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data/testing/minimal_curvature/ -I /workspace/data/final_data/testing/maximal_curvature/ -I /workspace/data/final_data/testing/profile_curvature/ -I /workspace/data/final_data/testing/stdon/ -I /workspace/data/final_data/testing/multiscale_stdon/ -I /workspace/data/final_data/testing/elevation_above_pit/ -I /workspace/data/final_data/testing/depthinsink/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed0_depth3/05m/combined1/trained.h5 /workspace/logfiles/seed0_depth3/05m/combined1/test.csv --classes=0,1 --depth=3

# SEED 1 depth3
echo "hillshade"
mkdir /workspace/logfiles/seed1_depth3/05m/hillshade1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/hillshade/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed1_depth3/05m/hillshade1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/hillshade/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed1_depth3/05m/hillshade1/trained.h5 /workspace/logfiles/seed1_depth3/05m/hillshade1/test.csv --classes=0,1 --depth=3

echo "maximum elevation deviation"
mkdir /workspace/logfiles/seed1_depth3/05m/maxelevationdeviation1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/maxelevationdeviation/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed1_depth3/05m/maxelevationdeviation1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/maxelevationdeviation/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed1_depth3/05m/maxelevationdeviation1/trained.h5 /workspace/logfiles/seed1_depth3/05m/maxelevationdeviation1/test.csv --classes=0,1 --depth=3

echo "multiscale elevation percentile"
mkdir /workspace/logfiles/seed1_depth3/05m/multiscale_elevation_percentile1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/multiscaleelevationpercentile/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed1_depth3/05m/multiscale_elevation_percentile1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/multiscaleelevationpercentile/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed1_depth3/05m/multiscale_elevation_percentile1/trained.h5 /workspace/logfiles/seed1_depth3/05m/multiscale_elevation_percentile1/test.csv --classes=0,1 --depth=3

echo "minimal curvature"
mkdir /workspace/logfiles/seed1_depth3/05m/minimal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/minimal_curvature/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed1_depth3/05m/minimal_curvature1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/minimal_curvature/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed1_depth3/05m/minimal_curvature1/trained.h5 /workspace/logfiles/seed1_depth3/05m/minimal_curvature1/test.csv --classes=0,1 --depth=3

echo "maximal curvature"
mkdir /workspace/logfiles/seed1_depth3/05m/maximal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/maximal_curvature/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed1_depth3/05m/maximal_curvature1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/maximal_curvature/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed1_depth3/05m/maximal_curvature1/trained.h5 /workspace/logfiles/seed1_depth3/05m/maximal_curvature1/test.csv --classes=0,1 --depth=3

echo "profile curvature"
mkdir /workspace/logfiles/seed1_depth3/05m/profile_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/profile_curvature/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed1_depth3/05m/profile_curvature1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/profile_curvature/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed1_depth3/05m/profile_curvature1/trained.h5 /workspace/logfiles/seed1_depth3/05m/profile_curvature1/test.csv --classes=0,1 --depth=3

echo "spherical standard deviation of normal"
mkdir /workspace/logfiles/seed1_depth3/05m/spherical_standard_deviation_of_normal1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/stdon/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed1_depth3/05m/spherical_standard_deviation_of_normal1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/stdon/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed1_depth3/05m/spherical_standard_deviation_of_normal1/trained.h5 /workspace/logfiles/seed1_depth3/05m/spherical_standard_deviation_of_normal1/test.csv --classes=0,1 --depth=3

echo "multiscale_stdon"
mkdir /workspace/logfiles/seed1_depth3/05m/multiscale_stdon1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/multiscale_stdon/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed1_depth3/05m/multiscale_stdon1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/multiscale_stdon/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed1_depth3/05m/multiscale_stdon1/trained.h5 /workspace/logfiles/seed1_depth3/05m/multiscale_stdon1/test.csv --classes=0,1 --depth=3

echo "elevation above pit"
mkdir /workspace/logfiles/seed1_depth3/05m/elevation_above_pit1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/elevation_above_pit/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed1_depth3/05m/elevation_above_pit1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/elevation_above_pit/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed1_depth3/05m/elevation_above_pit1/trained.h5 /workspace/logfiles/seed1_depth3/05m/elevation_above_pit1/test.csv --classes=0,1 --depth=3

echo "depthinsink"
mkdir /workspace/logfiles/seed1_depth3/05m/depthinsink1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/depthinsink/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed1_depth3/05m/depthinsink1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/depthinsink/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed1_depth3/05m/depthinsink1/trained.h5 /workspace/logfiles/seed1_depth3/05m/depthinsink1/test.csv --classes=0,1 --depth=3

echo "everything combined"
mkdir /workspace/logfiles/seed1_depth3/05m/combined1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/hillshade/ -I /workspace/data/final_data/training/maxelevationdeviation/ -I /workspace/data/final_data/training/multiscaleelevationpercentile/ -I /workspace/data/final_data/training/minimal_curvature/ -I /workspace/data/final_data/training/maximal_curvature/ -I /workspace/data/final_data/training/profile_curvature/ -I /workspace/data/final_data/training/stdon/ -I /workspace/data/final_data/training/multiscale_stdon/ -I /workspace/data/final_data/training/elevation_above_pit/ -I /workspace/data/final_data/training/depthinsink/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed1_depth3/05m/combined1/ --weighting="mfb" --seed=1 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/hillshade/ -I /workspace/data/final_data/testing/maxelevationdeviation/ -I /workspace/data/final_data/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data/testing/minimal_curvature/ -I /workspace/data/final_data/testing/maximal_curvature/ -I /workspace/data/final_data/testing/profile_curvature/ -I /workspace/data/final_data/testing/stdon/ -I /workspace/data/final_data/testing/multiscale_stdon/ -I /workspace/data/final_data/testing/elevation_above_pit/ -I /workspace/data/final_data/testing/depthinsink/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed1_depth3/05m/combined1/trained.h5 /workspace/logfiles/seed1_depth3/05m/combined1/test.csv --classes=0,1 --depth=3

# seed2 depth3
echo "hillshade"
mkdir /workspace/logfiles/seed2_depth3/05m/hillshade1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/hillshade/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed2_depth3/05m/hillshade1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/hillshade/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed2_depth3/05m/hillshade1/trained.h5 /workspace/logfiles/seed2_depth3/05m/hillshade1/test.csv --classes=0,1 --depth=3

echo "maximum elevation deviation"
mkdir /workspace/logfiles/seed2_depth3/05m/maxelevationdeviation1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/maxelevationdeviation/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed2_depth3/05m/maxelevationdeviation1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/maxelevationdeviation/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed2_depth3/05m/maxelevationdeviation1/trained.h5 /workspace/logfiles/seed2_depth3/05m/maxelevationdeviation1/test.csv --classes=0,1 --depth=3

echo "multiscale elevation percentile"
mkdir /workspace/logfiles/seed2_depth3/05m/multiscale_elevation_percentile1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/multiscaleelevationpercentile/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed2_depth3/05m/multiscale_elevation_percentile1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/multiscaleelevationpercentile/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed2_depth3/05m/multiscale_elevation_percentile1/trained.h5 /workspace/logfiles/seed2_depth3/05m/multiscale_elevation_percentile1/test.csv --classes=0,1 --depth=3

echo "minimal curvature"
mkdir /workspace/logfiles/seed2_depth3/05m/minimal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/minimal_curvature/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed2_depth3/05m/minimal_curvature1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/minimal_curvature/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed2_depth3/05m/minimal_curvature1/trained.h5 /workspace/logfiles/seed2_depth3/05m/minimal_curvature1/test.csv --classes=0,1 --depth=3

echo "maximal curvature"
mkdir /workspace/logfiles/seed2_depth3/05m/maximal_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/maximal_curvature/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed2_depth3/05m/maximal_curvature1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/maximal_curvature/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed2_depth3/05m/maximal_curvature1/trained.h5 /workspace/logfiles/seed2_depth3/05m/maximal_curvature1/test.csv --classes=0,1 --depth=3

echo "profile curvature"
mkdir /workspace/logfiles/seed2_depth3/05m/profile_curvature1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/profile_curvature/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed2_depth3/05m/profile_curvature1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/profile_curvature/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed2_depth3/05m/profile_curvature1/trained.h5 /workspace/logfiles/seed2_depth3/05m/profile_curvature1/test.csv --classes=0,1 --depth=3

echo "spherical standard deviation of normal"
mkdir /workspace/logfiles/seed2_depth3/05m/spherical_standard_deviation_of_normal1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/stdon/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed2_depth3/05m/spherical_standard_deviation_of_normal1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/stdon/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed2_depth3/05m/spherical_standard_deviation_of_normal1/trained.h5 /workspace/logfiles/seed2_depth3/05m/spherical_standard_deviation_of_normal1/test.csv --classes=0,1 --depth=3

echo "multiscale_stdon"
mkdir /workspace/logfiles/seed2_depth3/05m/multiscale_stdon1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/multiscale_stdon/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed2_depth3/05m/multiscale_stdon1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/multiscale_stdon/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed2_depth3/05m/multiscale_stdon1/trained.h5 /workspace/logfiles/seed2_depth3/05m/multiscale_stdon1/test.csv --classes=0,1 --depth=3

echo "elevation above pit"
mkdir /workspace/logfiles/seed2_depth3/05m/elevation_above_pit1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/elevation_above_pit/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed2_depth3/05m/elevation_above_pit1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/elevation_above_pit/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed2_depth3/05m/elevation_above_pit1/trained.h5 /workspace/logfiles/seed2_depth3/05m/elevation_above_pit1/test.csv --classes=0,1 --depth=3

echo "depthinsink"
mkdir /workspace/logfiles/seed2_depth3/05m/depthinsink1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/depthinsink/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed2_depth3/05m/depthinsink1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/depthinsink/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed2_depth3/05m/depthinsink1/trained.h5 /workspace/logfiles/seed2_depth3/05m/depthinsink1/test.csv --classes=0,1 --depth=3

echo "everything combined"
mkdir /workspace/logfiles/seed2_depth3/05m/combined1/
python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data/training/hillshade/ -I /workspace/data/final_data/training/maxelevationdeviation/ -I /workspace/data/final_data/training/multiscaleelevationpercentile/ -I /workspace/data/final_data/training/minimal_curvature/ -I /workspace/data/final_data/training/maximal_curvature/ -I /workspace/data/final_data/training/profile_curvature/ -I /workspace/data/final_data/training/stdon/ -I /workspace/data/final_data/training/multiscale_stdon/ -I /workspace/data/final_data/training/elevation_above_pit/ -I /workspace/data/final_data/training/depthinsink/ /workspace/data/final_data/training/labels/ /workspace/logfiles/seed2_depth3/05m/combined1/ --weighting="mfb" --seed=2 --depth=3 --epochs=100 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data/testing/hillshade/ -I /workspace/data/final_data/testing/maxelevationdeviation/ -I /workspace/data/final_data/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data/testing/minimal_curvature/ -I /workspace/data/final_data/testing/maximal_curvature/ -I /workspace/data/final_data/testing/profile_curvature/ -I /workspace/data/final_data/testing/stdon/ -I /workspace/data/final_data/testing/multiscale_stdon/ -I /workspace/data/final_data/testing/elevation_above_pit/ -I /workspace/data/final_data/testing/depthinsink/ /workspace/data/final_data/testing/labels/ /workspace/logfiles/seed2_depth3/05m/combined1/trained.h5 /workspace/logfiles/seed2_depth3/05m/combined1/test.csv --classes=0,1 --depth=3
