#!/bin/bash 
echo "hillshade"
mkdir /workspace/logfiles/1m/hillshade1/
mkdir /workspace/logfiles/1m/hillshade2/
mkdir /workspace/logfiles/1m/hillshade3/
mkdir /workspace/logfiles/1m/hillshade4/
mkdir /workspace/logfiles/1m/hillshade5/

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/hillshade1/ --weighting="mfb" --seed=1 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/hillshade1/trained.h5 /workspace/logfiles/1m/hillshade1/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/hillshade2/ --weighting="mfb" --seed=2 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/hillshade2/trained.h5 /workspace/logfiles/1m/hillshade2/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/hillshade3/ --weighting="mfb" --seed=3 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/hillshade3/trained.h5 /workspace/logfiles/1m/hillshade3/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/hillshade4/ --weighting="mfb" --seed=4 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/hillshade4/trained.h5 /workspace/logfiles/1m/hillshade4/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/hillshade5/ --weighting="mfb" --seed=5 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/hillshade5/trained.h5 /workspace/logfiles/1m/hillshade5/test.csv --classes=0,1 --depth=2


echo "maxelevationdeviation"
mkdir /workspace/logfiles/1m/maxelevationdeviation1/
mkdir /workspace/logfiles/1m/maxelevationdeviation2/
mkdir /workspace/logfiles/1m/maxelevationdeviation3/
mkdir /workspace/logfiles/1m/maxelevationdeviation4/
mkdir /workspace/logfiles/1m/maxelevationdeviation5/

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maxelevationdeviation/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/maxelevationdeviation1/ --weighting="mfb" --seed=1 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/maxelevationdeviation1/trained.h5 /workspace/logfiles/1m/maxelevationdeviation1/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maxelevationdeviation/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/maxelevationdeviation2/ --weighting="mfb" --seed=2 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/maxelevationdeviation2/trained.h5 /workspace/logfiles/1m/maxelevationdeviation2/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maxelevationdeviation/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/maxelevationdeviation3/ --weighting="mfb" --seed=3 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/maxelevationdeviation3/trained.h5 /workspace/logfiles/1m/maxelevationdeviation3/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maxelevationdeviation/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/maxelevationdeviation4/ --weighting="mfb" --seed=4 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/maxelevationdeviation4/trained.h5 /workspace/logfiles/1m/maxelevationdeviation4/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maxelevationdeviation/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/maxelevationdeviation5/ --weighting="mfb" --seed=5 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/maxelevationdeviation5/trained.h5 /workspace/logfiles/1m/maxelevationdeviation5/test.csv --classes=0,1 --depth=2


echo "multiscaleelevationpercentile"
mkdir /workspace/logfiles/1m/multiscaleelevationpercentile1/
mkdir /workspace/logfiles/1m/multiscaleelevationpercentile2/
mkdir /workspace/logfiles/1m/multiscaleelevationpercentile3/
mkdir /workspace/logfiles/1m/multiscaleelevationpercentile4/
mkdir /workspace/logfiles/1m/multiscaleelevationpercentile5/

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/multiscaleelevationpercentile1/ --weighting="mfb" --seed=1 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/multiscaleelevationpercentile1/trained.h5 /workspace/logfiles/1m/multiscaleelevationpercentile1/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/multiscaleelevationpercentile2/ --weighting="mfb" --seed=2 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/multiscaleelevationpercentile2/trained.h5 /workspace/logfiles/1m/multiscaleelevationpercentile2/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/multiscaleelevationpercentile3/ --weighting="mfb" --seed=3 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/multiscaleelevationpercentile3/trained.h5 /workspace/logfiles/1m/multiscaleelevationpercentile3/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/multiscaleelevationpercentile4/ --weighting="mfb" --seed=4 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/multiscaleelevationpercentile4/trained.h5 /workspace/logfiles/1m/multiscaleelevationpercentile4/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/multiscaleelevationpercentile5/ --weighting="mfb" --seed=5 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/multiscaleelevationpercentile5/trained.h5 /workspace/logfiles/1m/multiscaleelevationpercentile5/test.csv --classes=0,1 --depth=2

echo "training/minimal_curvature"
mkdir /workspace/logfiles/1m/minimal_curvature1/
mkdir /workspace/logfiles/1m/minimal_curvature2/
mkdir /workspace/logfiles/1m/minimal_curvature3/
mkdir /workspace/logfiles/1m/minimal_curvature4/
mkdir /workspace/logfiles/1m/minimal_curvature5/

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/minimal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/minimal_curvature1/ --weighting="mfb" --seed=1 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/minimal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/minimal_curvature1/trained.h5 /workspace/logfiles/1m/minimal_curvature1/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/minimal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/minimal_curvature2/ --weighting="mfb" --seed=2 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/minimal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/minimal_curvature2/trained.h5 /workspace/logfiles/1m/minimal_curvature2/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/minimal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/minimal_curvature3/ --weighting="mfb" --seed=3 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/minimal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/minimal_curvature3/trained.h5 /workspace/logfiles/1m/minimal_curvature3/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/minimal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/minimal_curvature4/ --weighting="mfb" --seed=4 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/minimal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/minimal_curvature4/trained.h5 /workspace/logfiles/1m/minimal_curvature4/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/minimal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/minimal_curvature5/ --weighting="mfb" --seed=5 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/minimal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/minimal_curvature5/trained.h5 /workspace/logfiles/1m/minimal_curvature5/test.csv --classes=0,1 --depth=2

echo "maximal_curvature"
mkdir /workspace/logfiles/1m/maximal_curvature1/
mkdir /workspace/logfiles/1m/maximal_curvature2/
mkdir /workspace/logfiles/1m/maximal_curvature3/
mkdir /workspace/logfiles/1m/maximal_curvature4/
mkdir /workspace/logfiles/1m/maximal_curvature5/

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maximal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/maximal_curvature1/ --weighting="mfb" --seed=1 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maximal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/maximal_curvature1/trained.h5 /workspace/logfiles/1m/maximal_curvature1/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maximal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/maximal_curvature2/ --weighting="mfb" --seed=2 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maximal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/maximal_curvature2/trained.h5 /workspace/logfiles/1m/maximal_curvature2/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maximal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/maximal_curvature3/ --weighting="mfb" --seed=3 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maximal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/maximal_curvature3/trained.h5 /workspace/logfiles/1m/maximal_curvature3/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maximal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/maximal_curvature4/ --weighting="mfb" --seed=4 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maximal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/maximal_curvature4/trained.h5 /workspace/logfiles/1m/maximal_curvature4/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/maximal_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/maximal_curvature5/ --weighting="mfb" --seed=5 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/maximal_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/maximal_curvature5/trained.h5 /workspace/logfiles/1m/maximal_curvature5/test.csv --classes=0,1 --depth=2

echo "profile_curvature"
mkdir /workspace/logfiles/1m/profile_curvature1/
mkdir /workspace/logfiles/1m/profile_curvature2/
mkdir /workspace/logfiles/1m/profile_curvature3/
mkdir /workspace/logfiles/1m/profile_curvature4/
mkdir /workspace/logfiles/1m/profile_curvature5/

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/profile_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/profile_curvature1/ --weighting="mfb" --seed=1 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/profile_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/profile_curvature1/trained.h5 /workspace/logfiles/1m/profile_curvature1/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/profile_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/profile_curvature2/ --weighting="mfb" --seed=2 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/profile_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/profile_curvature2/trained.h5 /workspace/logfiles/1m/profile_curvature2/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/profile_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/profile_curvature3/ --weighting="mfb" --seed=3 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/profile_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/profile_curvature3/trained.h5 /workspace/logfiles/1m/profile_curvature3/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/profile_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/profile_curvature4/ --weighting="mfb" --seed=4 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/profile_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/profile_curvature4/trained.h5 /workspace/logfiles/1m/profile_curvature4/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/profile_curvature/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/profile_curvature5/ --weighting="mfb" --seed=5 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/profile_curvature/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/profile_curvature5/trained.h5 /workspace/logfiles/1m/profile_curvature5/test.csv --classes=0,1 --depth=2

echo "stdon"
mkdir /workspace/logfiles/1m/stdon1/
mkdir /workspace/logfiles/1m/stdon2/
mkdir /workspace/logfiles/1m/stdon3/
mkdir /workspace/logfiles/1m/stdon4/
mkdir /workspace/logfiles/1m/stdon5/

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/stdon1/ --weighting="mfb" --seed=1 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/stdon1/trained.h5 /workspace/logfiles/1m/stdon1/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/stdon2/ --weighting="mfb" --seed=2 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/stdon2/trained.h5 /workspace/logfiles/1m/stdon2/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/stdon3/ --weighting="mfb" --seed=3 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/stdon3/trained.h5 /workspace/logfiles/1m/stdon3/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/stdon4/ --weighting="mfb" --seed=4 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/stdon4/trained.h5 /workspace/logfiles/1m/stdon4/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/stdon5/ --weighting="mfb" --seed=5 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/stdon5/trained.h5 /workspace/logfiles/1m/stdon5/test.csv --classes=0,1 --depth=2

echo "multiscale_stdon"
mkdir /workspace/logfiles/1m/multiscale_stdon1/
mkdir /workspace/logfiles/1m/multiscale_stdon2/
mkdir /workspace/logfiles/1m/multiscale_stdon3/
mkdir /workspace/logfiles/1m/multiscale_stdon4/
mkdir /workspace/logfiles/1m/multiscale_stdon5/

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscale_stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/multiscale_stdon1/ --weighting="mfb" --seed=1 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/multiscale_stdon1/trained.h5 /workspace/logfiles/1m/multiscale_stdon1/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscale_stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/multiscale_stdon2/ --weighting="mfb" --seed=2 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/multiscale_stdon2/trained.h5 /workspace/logfiles/1m/multiscale_stdon2/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscale_stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/multiscale_stdon3/ --weighting="mfb" --seed=3 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/multiscale_stdon3/trained.h5 /workspace/logfiles/1m/multiscale_stdon3/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscale_stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/multiscale_stdon4/ --weighting="mfb" --seed=4 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/multiscale_stdon4/trained.h5 /workspace/logfiles/1m/multiscale_stdon4/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/multiscale_stdon/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/multiscale_stdon5/ --weighting="mfb" --seed=5 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/multiscale_stdon/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/multiscale_stdon5/trained.h5 /workspace/logfiles/1m/multiscale_stdon5/test.csv --classes=0,1 --depth=2

echo "elevation_above_pit"
mkdir /workspace/logfiles/1m/elevation_above_pit1/
mkdir /workspace/logfiles/1m/elevation_above_pit2/
mkdir /workspace/logfiles/1m/elevation_above_pit3/
mkdir /workspace/logfiles/1m/elevation_above_pit4/
mkdir /workspace/logfiles/1m/elevation_above_pit5/

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/elevation_above_pit/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/elevation_above_pit1/ --weighting="mfb" --seed=1 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/elevation_above_pit/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/elevation_above_pit1/trained.h5 /workspace/logfiles/1m/elevation_above_pit1/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/elevation_above_pit/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/elevation_above_pit2/ --weighting="mfb" --seed=2 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/elevation_above_pit/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/elevation_above_pit2/trained.h5 /workspace/logfiles/1m/elevation_above_pit2/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/elevation_above_pit/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/elevation_above_pit3/ --weighting="mfb" --seed=3 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/elevation_above_pit/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/elevation_above_pit3/trained.h5 /workspace/logfiles/1m/elevation_above_pit3/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/elevation_above_pit/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/elevation_above_pit4/ --weighting="mfb" --seed=4 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/elevation_above_pit/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/elevation_above_pit4/trained.h5 /workspace/logfiles/1m/elevation_above_pit4/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/elevation_above_pit/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/elevation_above_pit5/ --weighting="mfb" --seed=5 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/elevation_above_pit/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/elevation_above_pit5/trained.h5 /workspace/logfiles/1m/elevation_above_pit5/test.csv --classes=0,1 --depth=2


echo "depthinsink"
mkdir /workspace/logfiles/1m/depthinsink1/
mkdir /workspace/logfiles/1m/depthinsink2/
mkdir /workspace/logfiles/1m/depthinsink3/
mkdir /workspace/logfiles/1m/depthinsink4/
mkdir /workspace/logfiles/1m/depthinsink5/

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/depthinsink1/ --weighting="mfb" --seed=1 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/depthinsink1/trained.h5 /workspace/logfiles/1m/depthinsink1/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/depthinsink2/ --weighting="mfb" --seed=2 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/depthinsink2/trained.h5 /workspace/logfiles/1m/depthinsink2/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/depthinsink3/ --weighting="mfb" --seed=3 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/depthinsink3/trained.h5 /workspace/logfiles/1m/depthinsink3/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/depthinsink4/ --weighting="mfb" --seed=4 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/depthinsink4/trained.h5 /workspace/logfiles/1m/depthinsink4/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/depthinsink5/ --weighting="mfb" --seed=5 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/depthinsink5/trained.h5 /workspace/logfiles/1m/depthinsink5/test.csv --classes=0,1 --depth=2


mkdir /workspace/logfiles/1m/combined1/
mkdir /workspace/logfiles/1m/combined2/
mkdir /workspace/logfiles/1m/combined3/
mkdir /workspace/logfiles/1m/combined4/
mkdir /workspace/logfiles/1m/combined5/

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ -I /workspace/data/final_data_1m/training/maxelevationdeviation/ -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/training/minimal_curvature/ -I /workspace/data/final_data_1m/training/maximal_curvature/ -I /workspace/data/final_data_1m/training/profile_curvature/ -I /workspace/data/final_data_1m/training/stdon/ -I /workspace/data/final_data_1m/training/multiscale_stdon/ -I /workspace/data/final_data_1m/training/elevation_above_pit/ -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/combined1 --weighting="mfb" --seed=1 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/testing/minimal_curvature/ -I /workspace/data/final_data_1m/testing/maximal_curvature/ -I /workspace/data/final_data_1m/testing/profile_curvature/ -I /workspace/data/final_data_1m/testing/stdon/ -I /workspace/data/final_data_1m/testing/multiscale_stdon/ -I /workspace/data/final_data_1m/testing/elevation_above_pit/ -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/combined1/trained.h5 /workspace/logfiles/1m/combined1/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ -I /workspace/data/final_data_1m/training/maxelevationdeviation/ -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/training/minimal_curvature/ -I /workspace/data/final_data_1m/training/maximal_curvature/ -I /workspace/data/final_data_1m/training/profile_curvature/ -I /workspace/data/final_data_1m/training/stdon/ -I /workspace/data/final_data_1m/training/multiscale_stdon/ -I /workspace/data/final_data_1m/training/elevation_above_pit/ -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/combined2 --weighting="mfb" --seed=2 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/testing/minimal_curvature/ -I /workspace/data/final_data_1m/testing/maximal_curvature/ -I /workspace/data/final_data_1m/testing/profile_curvature/ -I /workspace/data/final_data_1m/testing/stdon/ -I /workspace/data/final_data_1m/testing/multiscale_stdon/ -I /workspace/data/final_data_1m/testing/elevation_above_pit/ -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/combined2/trained.h5 /workspace/logfiles/1m/combined2/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ -I /workspace/data/final_data_1m/training/maxelevationdeviation/ -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/training/minimal_curvature/ -I /workspace/data/final_data_1m/training/maximal_curvature/ -I /workspace/data/final_data_1m/training/profile_curvature/ -I /workspace/data/final_data_1m/training/stdon/ -I /workspace/data/final_data_1m/training/multiscale_stdon/ -I /workspace/data/final_data_1m/training/elevation_above_pit/ -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/combined3 --weighting="mfb" --seed=3 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/testing/minimal_curvature/ -I /workspace/data/final_data_1m/testing/maximal_curvature/ -I /workspace/data/final_data_1m/testing/profile_curvature/ -I /workspace/data/final_data_1m/testing/stdon/ -I /workspace/data/final_data_1m/testing/multiscale_stdon/ -I /workspace/data/final_data_1m/testing/elevation_above_pit/ -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/combined3/trained.h5 /workspace/logfiles/1m/combined3/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ -I /workspace/data/final_data_1m/training/maxelevationdeviation/ -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/training/minimal_curvature/ -I /workspace/data/final_data_1m/training/maximal_curvature/ -I /workspace/data/final_data_1m/training/profile_curvature/ -I /workspace/data/final_data_1m/training/stdon/ -I /workspace/data/final_data_1m/training/multiscale_stdon/ -I /workspace/data/final_data_1m/training/elevation_above_pit/ -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/combined4 --weighting="mfb" --seed=4 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/testing/minimal_curvature/ -I /workspace/data/final_data_1m/testing/maximal_curvature/ -I /workspace/data/final_data_1m/testing/profile_curvature/ -I /workspace/data/final_data_1m/testing/stdon/ -I /workspace/data/final_data_1m/testing/multiscale_stdon/ -I /workspace/data/final_data_1m/testing/elevation_above_pit/ -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/combined4/trained.h5 /workspace/logfiles/1m/combined4/test.csv --classes=0,1 --depth=2

python /workspace/code/semantic_segmentation/train_unet.py -I /workspace/data/final_data_1m/training/hillshade/ -I /workspace/data/final_data_1m/training/maxelevationdeviation/ -I /workspace/data/final_data_1m/training/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/training/minimal_curvature/ -I /workspace/data/final_data_1m/training/maximal_curvature/ -I /workspace/data/final_data_1m/training/profile_curvature/ -I /workspace/data/final_data_1m/training/stdon/ -I /workspace/data/final_data_1m/training/multiscale_stdon/ -I /workspace/data/final_data_1m/training/elevation_above_pit/ -I /workspace/data/final_data_1m/training/depthinsink/ /workspace/data/final_data_1m/training/labels/ /workspace/logfiles/1m/combined5 --weighting="mfb" --seed=5 --depth=2 --epochs=200 --batch_size=16 --classes=0,1
python /workspace/code/semantic_segmentation/evaluate_unet.py -I /workspace/data/final_data_1m/testing/hillshade/ -I /workspace/data/final_data_1m/testing/maxelevationdeviation/ -I /workspace/data/final_data_1m/testing/multiscaleelevationpercentile/ -I /workspace/data/final_data_1m/testing/minimal_curvature/ -I /workspace/data/final_data_1m/testing/maximal_curvature/ -I /workspace/data/final_data_1m/testing/profile_curvature/ -I /workspace/data/final_data_1m/testing/stdon/ -I /workspace/data/final_data_1m/testing/multiscale_stdon/ -I /workspace/data/final_data_1m/testing/elevation_above_pit/ -I /workspace/data/final_data_1m/testing/depthinsink/ /workspace/data/final_data_1m/testing/labels/ /workspace/logfiles/1m/combined5/trained.h5 /workspace/logfiles/1m/combined5/test.csv --classes=0,1 --depth=2




