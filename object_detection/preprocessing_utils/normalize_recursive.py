'''
This code converts the float32 pixel values in tif-files from 0.0-1.0 to 0.0-255.0. This is because the 
object detector by default expecting pixel values between 0 and 255.
'''
import os
from tifffile import imread, imwrite
import pdb
import numpy as np
import logging
try:
    import gdal
except:
    from osgeo import gdal

def copy_geotransform_metadata(src_path, dst_path):
    # Open source GeoTIFF file
    src_dataset = gdal.Open(src_path, gdal.GA_ReadOnly)
    if src_dataset is None:
        raise Exception(f"Unable to open source GeoTIFF file: {src_path}")

    # Get geographical metadata
    geotransform = src_dataset.GetGeoTransform()
    projection = src_dataset.GetProjection()

    # Close source dataset
    src_dataset = None

    # Open destination GeoTIFF file
    dst_dataset = gdal.Open(dst_path, gdal.GA_Update)
    if dst_dataset is None:
        raise Exception(f"Unable to open destination GeoTIFF file: {dst_path}")

    # Set the geotransform and projection on the destination dataset
    dst_dataset.SetGeoTransform(geotransform)
    dst_dataset.SetProjection(projection)

    # Close destination dataset
    dst_dataset = None

    print(f"Successfully copied metadata from {src_path} to {dst_path}")


def normalize(source_path, destination_path):
    search_dir = source_path

    for root, dirs, files in os.walk(search_dir):
        print('Processing: root {}, dir {}, files {}'.format(root, dir, files))
        for f in files:
            if f.endswith('.tif'):
                image_path = os.path.join(root, f)
                print(f'Reading image {image_path}')
                img = imread(image_path)
                normalised = img * 255
                output_path = root.replace(source_path, destination_path)

                # Check if path needs to be created.
                if not os.path.exists(output_path):
                    print(f'Creating dir {output_path}')
                    os.makedirs(output_path, exist_ok=True)

                # Write file
                print(f'Writing file {os.path.join(output_path, f)}')
                imwrite(os.path.join(output_path, f), normalised)
                copy_geotransform_metadata(image_path, os.path.join(output_path, f))


source = 'datasets/topographical_indicies/'
destination = 'datasets/topographical_indicies_normalized/'

normalize(source, destination)

