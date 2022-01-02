import os
from tifffile import tifffile
from shutil import copyfile
import numpy as np
def main(image_path, label_path, outhput_image_path, numpixels, output_label_path):
    for chip in os.listdir(image_path):
        if chip.endswith('.tif'):
            
            imagewithpath = image_path + chip
            labelwithpath = label_path + chip

            image = tifffile.imread(labelwithpath)
            tilesum = np.sum(image)

            copied_image = outhput_image_path + chip
            copied_labels = output_label_path + chip

            if tilesum > numpixels:
                copyfile(imagewithpath, copied_image)
                copyfile(labelwithpath, copied_labels)
                # print(chip,' have ', tilesum,' labeled pixels')

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='copy chips with labeled pixels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('image_path', help='Path to image chips')
    parser.add_argument('label_path', help='path to label chips')
    parser.add_argument('outhput_image_path', help='outhput_path to copied image chips')
    parser.add_argument('numpixels', help = 'minimum number of pixels to copy files', type=int)
    parser.add_argument('output_label_path', help='outhput_path to copied labeled chips')

    args = vars(parser.parse_args())
    main(**args)
