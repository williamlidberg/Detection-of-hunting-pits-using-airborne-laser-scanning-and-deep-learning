#!/bin/bash 
echo "Randomly select 20 % of the chips to be moved to test data directories."
echo "A list of test chips will be located in /workspace/data/final_data_05m/test_chips.csv."
python /workspace/code/tools/create_partition.py /workspace/data/final_data_05m/training/labels/ /workspace/data/final_data_05m/test_chips.csv

echo "creating test directory"
mkdir /workspace/data/final_data_05m/testing

echo "empty old test directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/elevation_above_pit
mkdir /workspace/data/final_data_05m/testing/elevation_above_pit
echo "empty hillshade directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/hillshade
mkdir /workspace/data/final_data_05m/testing/hillshade
echo "empty labels directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/labels
mkdir /workspace/data/final_data_05m/testing/labels
echo "empty minimal_curvature directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/minimal_curvature
mkdir /workspace/data/final_data_05m/testing/minimal_curvature
echo "empty maximal_curvature directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/maximal_curvature
mkdir /workspace/data/final_data_05m/testing/maximal_curvature
echo "empty minimal_curvature directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/profile_curvature
mkdir /workspace/data/final_data_05m/testing/profile_curvature
echo "empty stdon directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/stdon
mkdir /workspace/data/final_data_05m/testing/stdon
echo "empty maxelevationdeviation directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/maxelevationdeviation
mkdir /workspace/data/final_data_05m/testing/maxelevationdeviation
echo "empty maxelevationdeviation directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/multiscaleelevationpercentile
mkdir /workspace/data/final_data_05m/testing/multiscaleelevationpercentile
echo "empty multiscaleelevationpercentile directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/depthinsink
mkdir /workspace/data/final_data_05m/testing/depthinsink
echo "empty depthinsink directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/maxelevationdeviation
mkdir /workspace/data/final_data_05m/testing/maxelevationdeviation
echo "empty maxelevationdeviation directories and creating new ones before moving test files"
rm -r /workspace/data/final_data_05m/testing/multiscale_stdon
mkdir /workspace/data/final_data_05m/testing/multiscale_stdon
echo "empty bounding_boxes dir"
rm -r /workspace/data/final_data_05m/testing/bounding_boxes
mkdir /workspace/data/final_data_05m/testing/bounding_boxes

echo "move test labels to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/labels/ /workspace/data/final_data_05m/testing/labels/ /workspace/data/final_data_05m/test_chips.csv

echo "move test hillshade to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/hillshade/ /workspace/data/final_data_05m/testing/hillshade/ /workspace/data/final_data_05m/test_chips.csv

echo "move test elevation_above_pit to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/elevation_above_pit/ /workspace/data/final_data_05m/testing/elevation_above_pit/ /workspace/data/final_data_05m/test_chips.csv

echo "move test Spherical Std Dev Of Normals to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/stdon/ /workspace/data/final_data_05m/testing/stdon/ /workspace/data/final_data_05m/test_chips.csv

echo "move test minimal_curvature to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/minimal_curvature/ /workspace/data/final_data_05m/testing/minimal_curvature/ /workspace/data/final_data_05m/test_chips.csv

echo "move test maximal_curvature to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/maximal_curvature/ /workspace/data/final_data_05m/testing/maximal_curvature/ /workspace/data/final_data_05m/test_chips.csv

echo "move test profile_curvature to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/profile_curvature/ /workspace/data/final_data_05m/testing/profile_curvature/ /workspace/data/final_data_05m/test_chips.csv

echo "move test maxelevationdeviation to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/maxelevationdeviation/ /workspace/data/final_data_05m/testing/maxelevationdeviation/ /workspace/data/final_data_05m/test_chips.csv

echo "move test multiscaleelevationpercentile to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/multiscaleelevationpercentile/ /workspace/data/final_data_05m/testing/multiscaleelevationpercentile/ /workspace/data/final_data_05m/test_chips.csv

echo "move test depthinsink to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/depthinsink/ /workspace/data/final_data_05m/testing/depthinsink/ /workspace/data/final_data_05m/test_chips.csv

echo "move test multiscale_stdon to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/multiscale_stdon/ /workspace/data/final_data_05m/testing/multiscale_stdon/ /workspace/data/final_data_05m/test_chips.csv

echo "move test bounding_boxes to new directories"
python /workspace/code/tools/partition_data.py /workspace/data/final_data_05m/training/bounding_boxes/ /workspace/data/final_data_05m/testing/bounding_boxes/ /workspace/data/final_data_05m/test_chips.csv