import os
import json
import logging
from PIL import Image

logging.basicConfig(level='INFO')
log = logging.getLogger()

def ensure_dir(directory: str):
    '''Takes a path. Creating it if it doesn't exist'''
    os.makedirs(directory, exist_ok=True)


def get_configuration(configuration: str) -> dict:
    '''Takes a path to a json configuration and returns it as a dict'''
    log.info("using config {}".format(configuration))
    with open('configurations/configurations.json', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data

def convert_jpg_to_tiff(source: str, destination: str):
    for root, dirs, filenames in os.walk(source):
        for f in filenames:
            im = Image.open(os.path.join(root,f))
            im.save(os.path.join(destination,f.replace('jpg','tiff')), 'TIFF')

def convert_tiff_to_jpg(source: str, destination: str):
    for root, dirs, filenames in os.walk(source):
        for f in filenames:
            if f.endswith('.tiff'):
                im = Image.open(os.path.join(root, f))
                im.save(os.path.join(destination, f.replace('.tiff', '.jpg')), 'JPEG')