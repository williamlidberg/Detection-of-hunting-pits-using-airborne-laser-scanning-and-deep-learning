#!/bin/bash

echo "empty old directories and creating new ones before splitting"
rm -r /workspace/data/lunar_data/split_data/hillshade/
mkdir /workspace/data/lunar_data/split_data/hillshade/
echo "Recreated hillshade"
rm -r /workspace/data/lunar_data/split_data/labels/
mkdir /workspace/data/lunar_data/split_data/labels/
echo "Recreated labels"
rm -r /workspace/data/lunar_data/split_data/minimal_curvature/
mkdir /workspace/data/lunar_data/split_data/minimal_curvature/
echo "Recreated min curvature"
rm -r /workspace/data/lunar_data/split_data/maximal_curvature/
mkdir /workspace/data/lunar_data/split_data/maximal_curvature/
echo "Recreated max curvature"
rm -r /workspace/data/lunar_data/split_data/profile_curvature/
mkdir /workspace/data/lunar_data/split_data/profile_curvature/
echo "Recreated profile curvature"
rm -r /workspace/data/lunar_data/split_data/stdon/
mkdir /workspace/data/lunar_data/split_data/stdon/
echo "Recreated stdon"
rm -r /workspace/data/lunar_data/split_data/multiscale_stdon/
mkdir /workspace/data/lunar_data/split_data/multiscale_stdon/
echo "Recreated multiscale stdon"
rm -r /workspace/data/lunar_data/split_data/maxelevationdeviation/
mkdir /workspace/data/lunar_data/split_data/maxelevationdeviation/
echo "Recreated max elevation deviation"
rm -r /workspace/data/lunar_data/split_data/multiscaleelevationpercentile/
mkdir /workspace/data/lunar_data/split_data/multiscaleelevationpercentile/
echo "Recreated max elevation multiscaleelevationpercentile"
rm -r /workspace/data/lunar_data/split_data/elevation_above_pit/
mkdir /workspace/data/lunar_data/split_data/elevation_above_pit/
echo "Recreated elevation above pit"
rm -r /workspace/data/lunar_data/split_data/depthinsink/
mkdir /workspace/data/lunar_data/split_data/depthinsink/
echo "Recreated depthinsink"
rm -r /workspace/data/lunar_data/split_data/object_labels/
mkdir /workspace/data/lunar_data/split_data/object_labels/
echo "Recreated object_labels"


echo "Split hillshade"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/hillshade/ /workspace/data/lunar_data/split_data/hillshade/ --tile_size 250
echo "Split elevation_above_pit"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/elevation_above_pits/ /workspace/data/lunar_data/split_data/elevation_above_pit/ --tile_size 250
echo "Split Spherical Std Dev Of Normals"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/stdon/ /workspace/data/lunar_data/split_data/stdon/ --tile_size 250
echo "Split minimal_curvature"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/minimal_curvature/ /workspace/data/lunar_data/split_data/minimal_curvature/ --tile_size 250
echo "Split profile_curvature"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/profile_curvature/ /workspace/data/lunar_data/split_data/profile_curvature/ --tile_size 250
echo "Split labels"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/labels/ /workspace/data/lunar_data/split_data/labels/ --tile_size 250
echo "Split maximal_curvature"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/maximal_curvature/ /workspace/data/lunar_data/split_data/maximal_curvature/ --tile_size 250
echo "Split multiscale_stdon"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/multiscale_stdon/ /workspace/data/lunar_data/split_data/multiscale_stdon/ --tile_size 250
echo "Split depthinsink"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/depthinsink/ /workspace/data/lunar_data/split_data/depthinsink/ --tile_size 250
echo "Split maxelevationdeviation"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/maxelevationdeviation/ /workspace/data/lunar_data/split_data/maxelevationdeviation/ --tile_size 250
echo "Split multiscaleelevationpercentile"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/multiscaleelevationpercentile/ /workspace/data/lunar_data/split_data/multiscaleelevationpercentile/ --tile_size 250
echo "Split object_labels"
python /workspace/code/tools/split_training_data.py /workspace/data/lunar_data/topographical_indices_normalized/object_labels/ /workspace/data/lunar_data/split_data/object_labels/ --tile_size 250