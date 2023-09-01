import urllib.request
import RestAPIs as ra
import time
import logging

logging.basicConfig(level='INFO')
log = logging.getLogger()


def get_image(coordinates: list, img_size: int, api: str, destination: str, name: str)->str:
    '''Takes a list that defines the coordinates of the image to download.
    Takes an integer that defines the image width and image height.
    Takes a string that is the key of the current configuration
    Take a string that defines the destination path of the image
    Takes a name that will be set to the image
    Returns the path of the downloaded image'''
    params_settings = ra.apis[api]['params'](coordinates, img_size)
    max_attempts = 6
    attempts = 0
    retry_delay = 20
    while attempts<max_attempts:
        attempts+=1
        try:
            image_url = ra.apis[api]['post'](params_settings)
            image_path = '{}/{}-{}-{}.tiff'.format(destination,coordinates,api, name)
            download_tif_image(image_url,image_path)
            return image_path
        except Exception as e:
            log.info(e)
            log.info('failed download image at attempt {}'.format(attempts))
            time.sleep(retry_delay)


def download_tif_image(tiff_image_url: str,destination_path: str):
    '''Takes the url to the image to download
    Takes a destination_path to download the image to'''
    urllib.request.urlretrieve(tiff_image_url, destination_path)