import argparse
import whitebox
wbt = whitebox.WhiteboxTools()

def laz_to_dem(laz_dir):
    """
    converts all laz tiles in a directory to a digital elevation raster.
    """  
    wbt.set_verbose_mode(True)
    wbt.set_working_dir(laz_dir)
    wbt.lidar_tin_gridding(parameter="elevation", 
    returns="last", # A DEM or DTM is usually obtained from the "last" returns, a DSM uses "first" returns (or better, use the lidar_digital_surface_model tool)
    resolution=0.5, # This is the spatial resolution of the output raster in meters and should depend on application needs and point density.
    exclude_cls= "0,1,3,4,5,6,7,8,10,11,12,13,14,15,16,17,18", # Example of classified points to be excluded from analysis i.e. class 9 is water.
    minz=None,
    maxz=None,
    max_triangle_edge_length=50
    )
    print("Completed TIN interpolation \n")


def main(laz_dir):
    laz_to_dem(laz_dir)

if __name__== '__main__':
    parser = argparse.ArgumentParser(
        description='Select the lidar tiles which contains training data',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('laz_dir', help='Path to directory where laz files are stored')    
    args = vars(parser.parse_args())
    main(**args)