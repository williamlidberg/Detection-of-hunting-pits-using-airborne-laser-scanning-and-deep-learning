import argparse
import sys
import os
import glob
import logging
import geopandas as gpd
sys.path.append("pipeline_utils")
import ImageRequest as ir
import ImageProcessing as ip
import Common as co
import YoloConverter as yc

def get_combined_image(coordinates: list, json_configuration: dict, configuration: str, img_size: int, destination_path: str):
    '''Download images from chosen data sources and combine them into a single tiff image.
    Takes a list of coordinates to represent a geo rectangle. Takes the dictionary of json configurations. Takes the configuration key
    Takes the image side resolution as an int. Takes a destination path to store the image'''
    channels = []
    #iterate throgh all APIS
    for i, api in enumerate(json_configuration[configuration]['apis']):
        image_path = ir.get_image(coordinates, img_size, api, 'feature_layers', 'image')
        channels.append(image_path)
        combined_image_path = '{}/combined_{}.tiff'.format(destination_path,coordinates)
    combined_image_path, number_of_channels = ip.combine_image_channels(channels, combined_image_path)


def main(opt):
    img_size, output_path, configuration, extents_path, meters_per_pixel, classes, split_only = opt.img_size, opt.output_path, opt.configuration, opt.extents, float(opt.meters_per_pixel), opt.classes, opt.split_only

    logging.basicConfig(level='INFO')
    log = logging.getLogger()
    log.info(f'Using config {configuration}')

    json_configuration = co.get_configuration(configuration)

    log.info(f'Creating dataset path {output_path}')
    co.ensure_dir(os.path.join(output_path, 'images'))
    co.ensure_dir(os.path.join(output_path, 'labels'))
    co.ensure_dir(os.path.join(output_path, 'split'))


    if(not split_only):
        log.info(f'Reading geo package file with bounding boxes in {extents_path}')
        extents = glob.glob(os.path.join(extents_path,"[!original_]*.gpkg"))

        #iterate through extent files
        for extents_index, extent_name in enumerate(extents):
            log.info(f'Creating data from extent {extent_name}')

            image_polygons = gpd.read_file(extent_name)
            image_polygons = image_polygons.set_crs(epsg=3006)
            
            original_polygons = gpd.read_file(os.path.dirname(extent_name)+'/original_'+os.path.basename(extent_name))
            #iterate through each polygon which represent 1 image
            for index, image_polygon in enumerate(image_polygons.iterrows()):
                coord = image_polygon[1]['geometry'].centroid.coords[0]
                offset = (img_size/2) * meters_per_pixel
                coordinates = [coord[0]-offset, coord[1]-offset, coord[0]+offset, coord[1]+offset]
                get_combined_image(coordinates, json_configuration, configuration, img_size, os.path.join(output_path,'images'))
                for original_polygon in original_polygons.iterrows():
                    if image_polygon[1].geometry.intersects(original_polygon[1].geometry):
                        xywh = yc.convert_polygons_to_yolo(image_polygon, original_polygon)
                        xywh = map(str,xywh)
                        xywh = " ".join(xywh)
                        with open(os.path.join(output_path,'labels','combined_{}.txt'.format(coordinates)), 'a') as file:
                            class_index = str(classes.index(original_polygon[1]['class']))
                            file.write(class_index + ' ' + xywh +'\n')
    
    log.info('create train and validation split')
    yc.split_yolo_data(output_path)
    yc.create_data_yaml(output_path, classes)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--output_path', action='store', default='datasets')
    parser.add_argument('--configuration', action='store', default='')
    parser.add_argument('--extents', action='store', default='')
    parser.add_argument('--meters_per_pixel', action='store', default=10)
    parser.add_argument('--classes', type=list, default=['obj'], help='the object classes. The order of the classes in this list will determine the yolo class index')
    parser.add_argument('--split_only', action='store', default=False, help='The split only option can be used if you only want to split a datset into train and validation subsets. The output will be in yolo-style data.yaml file.')

    opt = parser.parse_args()

    main(opt)