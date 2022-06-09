import os
import geopandas as gpd
import argparse
import shutil
import whitebox
wbt = whitebox.WhiteboxTools()


def copy_tiles(footprint, field_data, input_directory, output_directory):
    lidar_tiles_footprint = gpd.read_file(footprint)
    lidar_tiles_footprint.crs = '3006'
    field = gpd.read_file(field_data)
    intersect = gpd.sjoin(lidar_tiles_footprint, field, how='inner', op='intersects')
    uniqe_names = intersect['LAS_NM'].unique()
    print(len(uniqe_names), 'tiles intersected the field data')

    for name in os.listdir(input_directory):
        if name.endswith('.laz') and os.path.basename(name.replace('.laz','')) in uniqe_names:
            downladed_tile = input_directory + name
            copied_tile = output_directory + name
            shutil.copy(downladed_tile, copied_tile)


def main(footprint, field_data, input_directory, output_directory):
    copy_tiles(footprint, field_data, input_directory, output_directory)

if __name__== '__main__':
    parser = argparse.ArgumentParser(
        description='Select the lidar tiles which contains training data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('footprint', help='Path to tile index shapefile that contain a column with names of each indexruta')    
    parser.add_argument('field_data', help='shapefile containing field data')
   # parser.add_argument('aoi_polygon', help='shapefile containing area of interest')
    parser.add_argument('input_directory', help='input directory where all laz tiles are stored')
    parser.add_argument('output_directory', help = 'output directory for selected laz tiles')
    args = vars(parser.parse_args())
    main(**args)

