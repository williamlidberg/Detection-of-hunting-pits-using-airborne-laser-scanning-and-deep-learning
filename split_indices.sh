#!/bin/bash 
echo "empty old directories and creating new ones before splitting"
rm -r /workspace/data/split_data_pits/hillshade/
mkdir /workspace/data/split_data_pits/hillshade/
echo "Recreated hillshade"
rm -r /workspace/data/split_data_pits/labels/
mkdir /workspace/data/split_data_pits/labels/
echo "Recreated labels"
rm -r /workspace/data/split_data_pits/minimal_curvature/
mkdir /workspace/data/split_data_pits/minimal_curvature/
echo "Recreated min curvature"
rm -r /workspace/data/split_data_pits/maximal_curvature/
mkdir /workspace/data/split_data_pits/maximal_curvature/
echo "Recreated max curvature"
rm -r /workspace/data/split_data_pits/profile_curvature/
mkdir /workspace/data/split_data_pits/profile_curvature/
echo "Recreated profile curvature"
rm -r /workspace/data/split_data_pits/stdon/
mkdir /workspace/data/split_data_pits/stdon/
echo "Recreated stdon"
rm -r /workspace/data/split_data_pits/multiscale_stdon/
mkdir /workspace/data/split_data_pits/multiscale_stdon/
echo "Recreated multiscale stdon"
rm -r /workspace/data/split_data_pits/maxelevationdeviation/
mkdir /workspace/data/split_data_pits/maxelevationdeviation/
echo "Recreated max elevation deviation"
rm -r /workspace/data/split_data_pits/multiscaleelevationpercentile/
mkdir /workspace/data/split_data_pits/multiscaleelevationpercentile/
echo "Recreated max elevation multiscaleelevationpercentile"
rm -r /workspace/data/split_data_pits/elevation_above_pit/
mkdir /workspace/data/split_data_pits/elevation_above_pit/
echo "Recreated elevation above pit"
rm -r /workspace/data/split_data_pits/depthinsink/
mkdir /workspace/data/split_data_pits/depthinsink/
echo "Recreated depthinsink"


echo "Split hillshade"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/hillshade/ /workspace/data/split_data_pits/hillshade/ --tile_size 250
echo "Split elevation_above_pit"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/elevation_above_pit/ /workspace/data/split_data_pits/elevation_above_pit/ --tile_size 250
echo "Split Spherical Std Dev Of Normals"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/stdon/ /workspace/data/split_data_pits/stdon/ --tile_size 250
echo "Split minimal_curvature"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/minimal_curvature/ /workspace/data/split_data_pits/minimal_curvature/ --tile_size 250
echo "Split profile_curvature"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/profile_curvature/ /workspace/data/split_data_pits/profile_curvature/ --tile_size 250
echo "Split labels"
python /workspace/code/tools/split_training_data.py /workspace/data/segmentation_masks_pits/ /workspace/data/split_data_pits/labels/ --tile_size 250
echo "Split maximal_curvature"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/maximal_curvature/ /workspace/data/split_data_pits/maximal_curvature/ --tile_size 250
echo "Split multiscale_stdon"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/multiscale_stdon/ /workspace/data/split_data_pits/multiscale_stdon/ --tile_size 250
echo "Split depthinsink"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/depthinsink/ /workspace/data/split_data_pits/depthinsink/ --tile_size 250
echo "Split maxelevationdeviation"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/maxelevationdeviation/ /workspace/data/split_data_pits/maxelevationdeviation/ --tile_size 250
echo "Split multiscaleelevationpercentile"
python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/multiscaleelevationpercentile/ /workspace/data/split_data_pits/multiscaleelevationpercentile/ --tile_size 250


# echo "empty old directories and creating new ones before splitting"
# rm -r /workspace/data/split_data_pits/elevation_above_pit/
# mkdir /workspace/data/split_data_pits/elevation_above_pit/
# echo "Recreated elevation above pit"
# rm -r /workspace/data/split_data_pits/hillshade/
# mkdir /workspace/data/split_data_pits/hillshade/
# echo "Recreated hillshade"
# rm -r /workspace/data/split_data_pits/labels/
# mkdir /workspace/data/split_data_pits/labels/
# echo "Recreated labels"
# rm -r /workspace/data/split_data_pits/minimal_curvature/
# mkdir /workspace/data/split_data_pits/minimal_curvature/
# echo "Recreated min curvature"
# rm -r /workspace/data/split_data_pits/maximal_curvature/
# mkdir /workspace/data/split_data_pits/maximal_curvature/
# echo "Recreated max curvature"
# rm -r /workspace/data/split_data_pits/profile_curvature/
# mkdir /workspace/data/split_data_pits/profile_curvature/
# echo "Recreated profile curvature"
# rm -r /workspace/data/split_data_pits/stdon/
# mkdir /workspace/data/split_data_pits/stdon/
# echo "Recreated stdon"
# rm -r /workspace/data/split_data_pits/multiscale_stdon/
# mkdir /workspace/data/split_data_pits/multiscale_stdon/
# echo "Recreated multiscale stdon"
# rm -r /workspace/data/split_data_pits/maxelevationdeviation/
# mkdir /workspace/data/split_data_pits/maxelevationdeviation/
# echo "Recreated max elevation deviation"
# rm -r /workspace/data/split_data_pits/multiscaleelevationpercentile/
# mkdir /workspace/data/split_data_pits/multiscaleelevationpercentile/
# echo "Recreated max elevation multiscaleelevationpercentile"


# echo "Split hillshade"
#  python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/hillshade/ /workspace/data/split_data_pits/hillshade/ --tile_size 250
# echo "Split elevation_above_pit"
#  python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/elevation_above_pit/ /workspace/data/split_data_pits/elevation_above_pit/ --tile_size 250
# echo "Split Spherical Std Dev Of Normals"
#  python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/stdon/ /workspace/data/split_data_pits/stdon/ --tile_size 250
# echo "Split minimal_curvature"
#  python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/minimal_curvature/ /workspace/data/split_data_pits/minimal_curvature/ --tile_size 250
# echo "Split profile_curvature"
#  python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/profile_curvature/ /workspace/data/split_data_pits/profile_curvature/ --tile_size 250
# echo "Split labels"
#  python /workspace/code/tools/split_training_data.py /workspace/data/segmentation_masks_pits/ /workspace/data/split_data_pits/labels/ --tile_size 250
# echo "Split maximal_curvature"
#  python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/maximal_curvature/ /workspace/data/split_data_pits/maximal_curvature/ --tile_size 250
# echo "Split multiscale_stdon"
#  python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/multiscale_stdon/ /workspace/data/split_data_pits/multiscale_stdon/ --tile_size 250
# echo "Split depthinsink"
#  python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/depthinsink/ /workspace/data/split_data_pits/depthinsink/ --tile_size 250
# echo "Split maxelevationdeviation"
#  python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/maxelevationdeviation/ /workspace/data/split_data_pits/maxelevationdeviation/ --tile_size 250
# echo "Split multiscaleelevationpercentile"
#  python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/multiscaleelevationpercentile/ /workspace/data/split_data_pits/multiscaleelevationpercentile/ --tile_size 250
