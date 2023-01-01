#!/bin/bash 
echo "empty old directories and creating new ones before splitting"
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/elevation_above_pit/
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/elevation_above_pit/
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/hillshade/
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/hillshade/
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/labels/
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/labels/
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/minimal_curvature/
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/minimal_curvature/
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/profile_curvature/
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/profile_curvature/
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/stdon/
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/stdon/
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/maxelevationdeviation/
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/maxelevationdeviation/
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/multiscaleelevationpercentile/
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/multiscaleelevationpercentile/
rm -r /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/depthinsink/
mkdir /mnt/Extension_100TB/William/Projects/Cultural_remains/data/split_data_pits/depthinsink/

echo "Split labels"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/split_training_data.py /workspace/data/segmentation_masks_pits/ /workspace/data/split_data_pits/labels/ --tile_size 250
echo "Split hillshade"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/hillshade/ /workspace/data/split_data_pits/hillshade/ --tile_size 250
echo "Split elevation_above_pit"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/elevation_above_pit/ /workspace/data/split_data_pits/elevation_above_pit/ --tile_size 250
echo "Split Spherical Std Dev Of Normals"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/stdon/ /workspace/data/split_data_pits/stdon/ --tile_size 250
echo "Split minimal_curvature"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/minimal_curvature/ /workspace/data/split_data_pits/minimal_curvature/ --tile_size 250
echo "Split profile_curvature"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/profile_curvature/ /workspace/data/split_data_pits/profile_curvature/ --tile_size 250
echo "Split profile_curvature"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/maxelevationdeviation/ /workspace/data/split_data_pits/maxelevationdeviation/ --tile_size 250
echo "Split multiscaleelevationpercentile"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/multiscaleelevationpercentile/ /workspace/data/split_data_pits/multiscaleelevationpercentile/ --tile_size 250
echo "Split depthinsink"
docker run -v /mnt/Extension_100TB/William/GitHub/Remnants-of-charcoal-kilns:/workspace/code -v /mnt/Extension_100TB/William/Projects/Cultural_remains/data:/workspace/data -v /mnt/ramdisk:/workspace/temp -v /mnt/Extension_100TB/national_datasets/laserdataskog:/workspace/lidar segmentation:latest python /workspace/code/tools/split_training_data.py /workspace/data/topographical_indices_normalized_pits/depthinsink/ /workspace/data/split_data_pits/depthinsink/ --tile_size 250

