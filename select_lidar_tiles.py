import os
import argparse
import whitebox
wbt = whitebox.WhiteboxTools()

# example: python /workspace/code/select_lidar_tiles.py /workspace/lidar/pooled_laz_files/ /workspace/data/selected_lidar_tiles/ /workspace/data/charcoal_kilns/Merged_charcoal_kilns_william.shp
def lidar_tile_footprint(laz_dir):
    wbt.set_working_dir(laz_dir)
    wbt.lidar_tile_footprint(
        output=tile_index, 
        i=None, 
        hull=False
        )

def copy_tiles(tile_index, block_dir, pooled_laz_dir, copy_laz_dir):
    lidar_tiles_index = gpd.read_file(tile_index)
    path_to_block_metadata = block_dir + 'metadata/'
    json_list = []
    for tile in os.listdir(path_to_block_metadata):
        if tile.endswith('.json'):
            name = path_to_block_metadata + tile
            tile_json = gpd.read_file(name)
            json_list.append(tile_json)
    # merge all json polygons to a geopandas dataframe        
    block_extent = gpd.GeoDataFrame(pd.concat(json_list, ignore_index=True), crs=json_list[0].crs)
    # intersecct lidar tiles with block extent with one tile overlap
    intersect = gpd.sjoin(lidar_tiles_index, block_extent, how='inner', op='intersects')
    # get uniqe names
    uniqe_names = intersect['Indexruta'].unique()
    print(len(uniqe_names), 'tiles intersected the block')
    path_to_downloaded_data = pooled_laz_dir
    path_to_working_dir = copy_laz_dir
    names_relevant_tiles = []
    for name in os.listdir(path_to_downloaded_data):
        if name.endswith('.laz') and os.path.basename(name[7:20]) in uniqe_names:
            downladed_tile = path_to_downloaded_data + name
            copied_tile = path_to_working_dir + name
            shutil.copy(downladed_tile, copied_tile)
def select_tiles(input_directory, output_directory, aoi_polygon):
    if os.path.isdir(output_directory) != True: # Creates the output directory if it does not already exist
        os.mkdir(output_directory)

    wbt.select_tiles_by_polygon(
        indir=input_directory, 
        outdir=output_directory, 
        polygons=aoi_polygon
    )

def lidar_to_dem(input_laz_dir, output_dem_dir):
    if os.path.isdir(output_dem_dir) != True: # Creates the output directory if it does not already exist
        os.mkdir(output_dem_dir)

    wbt.set_working_dir(input_laz_dir, resolution)
    wbt.lidar_tin_gridding(parameter="elevation", 
        returns="last", # A DEM or DTM is usually obtained from the "last" returns, a DSM uses "first" returns (or better, use the lidar_digital_surface_model tool)
        resolution=resolution, # This is the spatial resolution of the output raster in meters and should depend on application needs and point density.
        exclude_cls= "0,1,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18", # Example of classified points to be excluded from analysis i.e. class 9 is water.
        minz=None,
        maxz=None,
        max_triangle_edge_length=50.0
    )
    print("Completed TIN interpolation \n")


def main(input_directory, output_laz_directory, aoi_polygon, output_dem_directory):  
    #lidar_tile_footprint(input_directory)
    select_tiles(input_directory, output_laz_directory, aoi_polygon)
    #lidar_to_dem(output_laz_directory, output_dem_directory)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                       description='Extract topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_directory', help='input directory where all laz tiles are stored')
    parser.add_argument('output_laz_directory', help = 'output directory for selected laz tiles')
    parser.add_argument('aoi_polygon', help = 'polygon over study area')
    parser.add_argument('output_dem_directory', help='output directory of dem files')
    parser.add_argument('--resolution',default=0.5, help='select output dem resolution. The default is 0.5 m')   
    args = vars(parser.parse_args())
    main(**args)