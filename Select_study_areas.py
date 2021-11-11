import argparse
import geopandas as gpd
import shapely.speedups
shapely.speedups.enable()
import rasterio

parser = argparse.ArgumentParser(description='Identify relevant lidar tiles and copies coresponding dem to new directory. ')

def main(field_data_shapefile_path, shape_tile_path, input_tile_path, output_tile_path):
    field_data = gpd.read_file(field_data_shapefile_path)
    lidar_tiles = gpd.read(shape_tile_path)
    intersect = gpd.sjoin(field_data, lidar_tiles[['name', 'geometry']], how = 'left', op = 'intersects')
    list_of_relevant_lidar_tiles = intersect['names'].values.tolist()
    for name in list_of_relevant_lidar_tiles:
        input_tile = input_tile_path + name + '.tif'
        output_tile = output_tile_path + name + '.tif'
        rasterio.shutil.copy(input_tile, output_tile)



def copy_tiles (input_path, output_path):
    rasterio.shutil.copy(input_path, output_path)


if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='Select the lidar tiles which contains training data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('field_data_shapefile_path', help='Path to shapefile that contains training data')
    parser.add_argument('shape_tile_path', help='input shapefile with lidar tiles')
    parser.add_argument('input_tile_path', help='input_tile_path')
    parser.add_argument('output_tile_path', help='output_tile_path')
    args = vars(parser.parse_args())
    main(**args)
    
