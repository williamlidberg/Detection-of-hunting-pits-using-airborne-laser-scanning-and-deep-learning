from tifffile import imread
import numpy as np
import logging
try:
    import gdal
except:
    from osgeo import gdal
import exifread
import RestAPIs as ra
import logging

logging.basicConfig(level='INFO')
log = logging.getLogger()


def read_tiff_data(image_path: str, selected_tags: str) -> list:
    '''Takes a string of the image path. Takes ids of exif tag. returns the values of the exif tags'''
    tag_ids = {
        'meter_per_pixel': 'Image Tag 0x830E',
        'gps_coordinates': 'Image Tag 0x8482'
    }
    tags = exifread.process_file(open(image_path, 'rb'))
    results = []
    for tag in selected_tags:
        results.append(tags[tag_ids[tag]].values)

    return results


def write_geotif_at_location(ref_image_filepath: str, out_image_filepath: str, list_of_numpy_arr: np.ndarray):
    """

    Writes a geotif at the same position as a reference image. 
    Each band in the geotif is added in the list as np.array 
    
    input:
        ref_image_filepath (string) - path to georeferences image
        out_image_filepath (string) - path to output image
        list_of_numpy_arr (list)  - list of 2d nparrys, shape should be of same size as shape of ref_image_filepath
    output:
        None
    """
    logging.info(f'Writing geotif {out_image_filepath}')
    ds = gdal.Open(ref_image_filepath)
    band = ds.GetRasterBand(1)
    arr = band.ReadAsArray()
    [rows, cols] = arr.shape
    
    driver = gdal.GetDriverByName("GTiff")
   # outdata = driver.Create(out_image_filepath, cols, rows, len(list_of_numpy_arr), gdal.GDT_Float32, options=['COMPRESS=DEFLATE'])
    outdata = driver.Create(out_image_filepath, cols, rows, len(list_of_numpy_arr), gdal.GDT_Byte, options=['COMPRESS=LZW'])

    outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input

    for i in range(len(list_of_numpy_arr)):
        outdata.GetRasterBand(i+1).WriteArray(list_of_numpy_arr[i])
        outdata.GetRasterBand(i+1).SetNoDataValue(10000)##if you want these values transparent
    outdata.FlushCache() ##saves to disk!!
    outdata = None
    band = None
    ds = None
    return None


def combine_image_channels(channels: list, destination_path: str) -> tuple:
    '''Takes a list of paths of images. Takes a destination path to store the combine image. Combines all images in channels into a single tiff file.
    Returns the destination path to the image and the number of channels to the combined image, to be able to check if the provided image is compatible to the AI model'''
    imgs = []
    for i, img in enumerate(channels):
        im = imread(img)
        logging.info(f'Reading file {img}')
        #normalize:
        #image name is named: [coordinates]-feature_name-example_name:
        feature_name = img.split('-')[-2].split('.')[0]
        #point to the normalization method with the feature_name as the key:
        logging.info(f'normalizing for {feature_name}')
        im = ra.apis[feature_name]['normalization'](im)
        if(len(im.shape) > 2):
            #Channel order has to switch for multi channel images
            im = im.transpose(2,0,1)
            for chan in im:
                imgs.append(chan)
        else:
            #Grayscale image doesn't have to change channel order
            imgs.append(im)
    imgs = np.array(imgs)
    if len(imgs) > 0:
        try:
            write_geotif_at_location(channels[0],destination_path, imgs)
            return destination_path, len(imgs)
        except Exception as e:

            log.info(e)
            log.info(i) 