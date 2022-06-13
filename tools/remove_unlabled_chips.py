import os
from tifffile import tifffile
import numpy as np
# , image_path_4, image_path_5, image_path_6, image_path_7, image_path_8, image_path_9, image_path_10)
def main(numpixels, label_path, image_path_1, image_path_2, image_path_3, image_path_4):
    for chip in os.listdir(label_path):
        if chip.endswith('.tif'):          
            labelwithpath = label_path + chip
            imagewithpath_1 = image_path_1 + chip
            imagewithpath_2 = image_path_2 + chip
            imagewithpath_3 = image_path_3 + chip
            imagewithpath_4 = image_path_4 + chip
            # imagewithpath_5 = image_path_5 + chip
            # imagewithpath_6 = image_path_6 + chip
            # imagewithpath_7 = image_path_7 + chip
            # imagewithpath_8 = image_path_8 + chip
            # imagewithpath_9 = image_path_9 + chip
            # imagewithpath_10 = image_path_10 + chip
         
            image = tifffile.imread(labelwithpath)
            tilesum = np.sum(image)
            if tilesum < numpixels:
                os.remove(labelwithpath)
                os.remove(imagewithpath_1)
                os.remove(imagewithpath_2)
                os.remove(imagewithpath_3)
                os.remove(imagewithpath_4)
                # os.remove(imagewithpath_5)
                # os.remove(imagewithpath_6)
                # os.remove(imagewithpath_7)
                # os.remove(imagewithpath_8)
                # os.remove(imagewithpath_9)
                # os.remove(imagewithpath_10)
                #print('removed ', chip, 'and all related images')

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='copy chips with labeled pixels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('numpixels', help = 'minimum number of pixels to copy files', type=int)
    parser.add_argument('label_path', help='path to label chips')
    parser.add_argument('image_path_1', help='dir of image path 1')
    parser.add_argument('image_path_2', help='dir of image path 2')
    parser.add_argument('image_path_3', help='dir of image path 3')
    parser.add_argument('image_path_4', help='dir of image path 3')
    # parser.add_argument('image_path_5', help='dir of image path 3')
    # parser.add_argument('image_path_6',help='dir of image path 3')
    # parser.add_argument('image_path_7',help='dir of image path 3')
    # parser.add_argument('image_path_8',help='dir of image path 3')
    # parser.add_argument('image_path_9',help='dir of image path 3')
    # parser.add_argument('image_path_10',help='dir of image path 3')



    args = vars(parser.parse_args())
    main(**args)