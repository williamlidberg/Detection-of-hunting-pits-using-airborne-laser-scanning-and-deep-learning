import os
from tifffile import tifffile
import numpy as np
import shutil


def main(original_dir, box_dir):
    strange_boxes = os.listdir(box_dir)

    for subdir, dirs, files in os.walk(original_dir):
        for file in files:
            if file in strange_boxes:
                os.remove(os.path.join(subdir, file))

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='copy chips with labeled pixels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('original_dir', help='path to label chips')
    parser.add_argument('box_dir', help = 'path to correct_bounding boxes')



    args = vars(parser.parse_args())
    main(**args)