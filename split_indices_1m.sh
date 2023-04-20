#!/bin/bash 
echo "empty old directories and creating new ones before splitting"
rm -r /workspace/data/split_data_pits_1m/hillshade/
mkdir /workspace/data/split_data_pits_1m/hillshade/
echo "Recreated hillshade"
rm -r /workspace/data/split_data_pits_1m/labels/
mkdir /workspace/data/split_data_pits_1m/labels/
echo "Recreated labels"
rm -r /workspace/data/split_data_pits_1m/minimal_curvature/
mkdir /workspace/data/split_data_pits_1m/minimal_curvature/
echo "Recreated min curvature"
rm -r /workspace/data/split_data_pits_1m/maximal_curvature/
mkdir /workspace/data/split_data_pits_1m/maximal_curvature/
echo "Recreated max curvature"
rm -r /workspace/data/split_data_pits_1m/profile_curvature/
mkdir /workspace/data/split_data_pits_1m/profile_curvature/
echo "Recreated profile curvature"
rm -r /workspace/data/split_data_pits_1m/stdon/
mkdir /workspace/data/split_data_pits_1m/stdon/
echo "Recreated stdon"
rm -r /workspace/data/split_data_pits_1m/multiscale_stdon/
mkdir /workspace/data/split_data_pits_1m/multiscale_stdon/
echo "Recreated multiscale stdon"
rm -r /workspace/data/split_data_pits_1m/maxelevationdeviation/
mkdir /workspace/data/split_data_pits_1m/maxelevationdeviation/
echo "Recreated max elevation deviation"
rm -r /workspace/data/split_data_pits_1m/multiscaleelevationpercentile/
mkdir /workspace/data/split_data_pits_1m/multiscaleelevationpercentile/
echo "Recreated max elevation multiscaleelevationpercentile"
rm -r /workspace/data/split_data_pits_1m/elevation_above_pit/
mkdir /workspace/data/split_data_pits_1m/elevation_above_pit/
echo "Recreated elevation above pit"
rm -r /workspace/data/split_data_pits_1m/depthinsink/
mkdir /workspace/data/split_data_pits_1m/depthinsink/
echo "Recreated depthinsink"


echo "Split hillshade"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits_1m/hillshade/ /workspace/data/split_data_pits_1m/hillshade/ --tile_size 250
echo "Split elevation_above_pit"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits_1m/elevation_above_pit/ /workspace/data/split_data_pits_1m/elevation_above_pit/ --tile_size 250
echo "Split Spherical Std Dev Of Normals"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits_1m/stdon/ /workspace/data/split_data_pits_1m/stdon/ --tile_size 250
echo "Split minimal_curvature"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits_1m/minimal_curvature/ /workspace/data/split_data_pits_1m/minimal_curvature/ --tile_size 250
echo "Split profile_curvature"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits_1m/profile_curvature/ /workspace/data/split_data_pits_1m/profile_curvature/ --tile_size 250
echo "Split labels"
python /workspace/code/tools/split_training_data.py /workspace/data/segmentation_masks_pits_1m/ /workspace/data/split_data_pits_1m/labels/ --tile_size 250
echo "Split maximal_curvature"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits_1m/maximal_curvature/ /workspace/data/split_data_pits_1m/maximal_curvature/ --tile_size 250
echo "Split multiscale_stdon"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits_1m/multiscale_stdon/ /workspace/data/split_data_pits_1m/multiscale_stdon/ --tile_size 250
echo "Split depthinsink"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits_1m/depthinsink/ /workspace/data/split_data_pits_1m/depthinsink/ --tile_size 250
echo "Split maxelevationdeviation"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits_1m/maxelevationdeviation/ /workspace/data/split_data_pits_1m/maxelevationdeviation/ --tile_size 250
echo "Split multiscaleelevationpercentile"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits_1m/multiscaleelevationpercentile/ /workspace/data/split_data_pits_1m/multiscaleelevationpercentile/ --tile_size 250
