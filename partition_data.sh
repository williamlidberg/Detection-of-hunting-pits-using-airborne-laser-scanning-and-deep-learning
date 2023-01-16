#!/bin/bash 
echo "Randomly select 20 % of the chips to be moved to test data directories."
echo "A list of test chips will be located in /workspace/data/test_chips.csv."
python /workspace/code/tools/create_partition.py /workspace/data/split_data_pits/labels/ /workspace/data/test_chips.csv

echo "empty old test directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/elevation_above_pit
mkdir /workspace/data/test_data_pits/elevation_above_pit
echo "empty hillshade directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/hillshade
mkdir /workspace/data/test_data_pits/hillshade
echo "empty labels directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/labels
mkdir /workspace/data/test_data_pits/labels
echo "empty minimal_curvature directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/minimal_curvature
mkdir /workspace/data/test_data_pits/minimal_curvature
echo "empty maximal_curvature directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/maximal_curvature
mkdir /workspace/data/test_data_pits/maximal_curvature
echo "empty minimal_curvature directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/profile_curvature
mkdir /workspace/data/test_data_pits/profile_curvature
echo "empty stdon directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/stdon
mkdir /workspace/data/test_data_pits/stdon
echo "empty maxelevationdeviation directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/maxelevationdeviation
mkdir /workspace/data/test_data_pits/maxelevationdeviation
echo "empty maxelevationdeviation directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/multiscaleelevationpercentile
mkdir /workspace/data/test_data_pits/multiscaleelevationpercentile
echo "empty multiscaleelevationpercentile directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/depthinsink
mkdir /workspace/data/test_data_pits/depthinsink
echo "empty depthinsink directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/maxelevationdeviation
mkdir /workspace/data/test_data_pits/maxelevationdeviation
echo "empty maxelevationdeviation directories and creating new ones before moving test files"
rm -r /workspace/data/test_data_pits/multiscale_stdon
mkdir /workspace/data/test_data_pits/multiscale_stdon


echo "move test labels to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/labels/ /workspace/data/test_data_pits/labels/ /workspace/data/test_chips.csv
echo "move test hillshade to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/hillshade/ /workspace/data/test_data_pits/hillshade/ /workspace/data/test_chips.csv
echo "move test elevation_above_pit to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/elevation_above_pit/ /workspace/data/test_data_pits/elevation_above_pit/ /workspace/data/test_chips.csv
echo "move test Spherical Std Dev Of Normals to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/stdon/ /workspace/data/test_data_pits/stdon/ /workspace/data/test_chips.csv
echo "move test minimal_curvature to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/minimal_curvature/ /workspace/data/test_data_pits/minimal_curvature/ /workspace/data/test_chips.csv
echo "move test maximal_curvature to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/maximal_curvature/ /workspace/data/test_data_pits/maximal_curvature/ /workspace/data/test_chips.csv
echo "move test profile_curvature to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/profile_curvature/ /workspace/data/test_data_pits/profile_curvature/ /workspace/data/test_chips.csv
echo "move test maxelevationdeviation to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/maxelevationdeviation/ /workspace/data/test_data_pits/maxelevationdeviation/ /workspace/data/test_chips.csv
echo "move test multiscaleelevationpercentile to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/multiscaleelevationpercentile/ /workspace/data/test_data_pits/multiscaleelevationpercentile/ /workspace/data/test_chips.csv
echo "move test depthinsink to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/depthinsink/ /workspace/data/test_data_pits/depthinsink/ /workspace/data/test_chips.csv
echo "move test multiscale_stdon to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/multiscale_stdon/ /workspace/data/test_data_pits/multiscale_stdon/ /workspace/data/test_chips.csv