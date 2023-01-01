#!/bin/bash 
echo "empty old directories and creating new ones before moving test files"
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/elevation_above_pit
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/elevation_above_pit
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/hillshade
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/hillshade
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/labels
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/labels
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/minimal_curvature
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/minimal_curvature
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/profile_curvature
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/profile_curvature
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/stdon
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/stdon
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/maxelevationdeviation
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/maxelevationdeviation
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/multiscaleelevationpercentile
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/multiscaleelevationpercentile
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/depthinsink
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/test_data_pits/depthinsink

echo "move test labels to new directories"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/labels/ /workspace/data/test_data_pits/labels/ /workspace/data/test_chips.csv
echo "move test hillshade to new directories"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/hillshade/ /workspace/data/test_data_pits/hillshade/ /workspace/data/test_chips.csv
echo "move test elevation_above_pit to new directories"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/elevation_above_pit/ /workspace/data/test_data_pits/elevation_above_pit/ /workspace/data/test_chips.csv
echo "move test Spherical Std Dev Of Normals to new directories"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/stdon/ /workspace/data/test_data_pits/stdon/ /workspace/data/test_chips.csv
echo "move test minimal_curvature to new directories"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/minimal_curvature/ /workspace/data/test_data_pits/minimal_curvature/ /workspace/data/test_chips.csv
echo "move test profile_curvature to new directories"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/profile_curvature/ /workspace/data/test_data_pits/profile_curvature/ /workspace/data/test_chips.csv
echo "move test maxelevationdeviation to new directories"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/maxelevationdeviation/ /workspace/data/test_data_pits/maxelevationdeviation/ /workspace/data/test_chips.csv
echo "move test multiscaleelevationpercentile to new directories"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/multiscaleelevationpercentile/ /workspace/data/test_data_pits/multiscaleelevationpercentile/ /workspace/data/test_chips.csv
echo "move test depthinsink to new directories"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/partition_data.py /workspace/data/split_data_pits/depthinsink/ /workspace/data/test_data_pits/depthinsink/ /workspace/data/test_chips.csv
