import os
import argparse

from attr import field
import whitebox
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()

def convert_polygon_segmentation(base_file_path, input_observations,field, output_label_path):
    for f in os.listdir(base_file_path):
        if f.endswith('.tif'):
            base = base_file_path + f
            label_tiles = output_label_path + f
            wbt.vector_polygons_to_raster(
                i = input_observations, 
                output = label_tiles, 
                field=field, 
                nodata=False, 
                cell_size=None, 
                base=base
            )

def main(base_file_path, input_observations, field, output_label_path):
    convert_polygon_segmentation(base_file_path, input_observations, field, output_label_path)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Convert vector observaions to binary raster labels',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('base_file_path', help='Path to input files to use as basefile')
    parser.add_argument('input_observations', help = 'shapefile with observations to convert to labels')
    parser.add_argument('field', help = 'select attribute to use for segmentation classes')
    parser.add_argument('output_label_path', help = 'path to output labels')
   

    args = vars(parser.parse_args())
    main(**args)