import os
import argparse
import tifffile
import numpy as np

def main(input_path):
    # count pixels
    list_0 = []
    list_1 = [] 
    for tile in os.listdir(input_path):
        if tile.endswith('.tif'):
            img_name = input_path + tile 
            chip = tifffile.imread(img_name)

            num_zeros = (chip == 0).sum()
            num_ones = (chip == 1).sum()
            list_0.append(num_zeros)
            list_1.append(num_ones)
            print('counted', (tile))
    total_0 = sum(list_0)
    total_1 = sum(list_1)
    print((total_1/total_0)*100,'% are labeled as 1')

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='counts the number of pixels labaled as 1.',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='path to directory of labels')

    args = vars(parser.parse_args())
    main(**args)