import whitebox
import argparse
wbt = whitebox.WhiteboxTools()

def main(laz_dir, footprint):
    wbt.set_working_dir(laz_dir)
    wbt.lidar_tile_footprint(
        output= footprint, 
        i=None, 
        hull=False
    )


if __name__== '__main__':
    parser = argparse.ArgumentParser(
        description='Select the lidar tiles which contains training data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('laz_dir', help='Path to tile index shapefile that contain a column with names of each indexruta')    
    parser.add_argument('footprint', help= 'path to shapefile with lidar footprint')
    args = vars(parser.parse_args())
    main(**args)