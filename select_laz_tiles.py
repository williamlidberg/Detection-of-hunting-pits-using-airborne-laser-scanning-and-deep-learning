# adapted from https://www.whiteboxgeo.com/manual/wbt_book/tutorials/lidar.html
import os
import argparse
import whitebox
wbt = whitebox.WhiteboxTools()


def main(input_directory, output_directory, aoi_polygon):
    wbt.set_verbose_mode(False) 
    
    if os.path.isdir(output_directory) != True: # Creates the output directory if it does not already exist
        os.mkdir(output_directory)

    wbt.select_tiles_by_polygon(
        indir=input_directory, 
        outdir=output_directory, 
        polygons=aoi_polygon
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                       description='Extract topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_directory', help='input directory where all laz tiles are stored')
    parser.add_argument('output_directory', help = 'output directory for selected laz tiles')
    parser.add_argument('aoi_polygon', help = 'polygon over study area')   
    args = vars(parser.parse_args())
    main(**args)