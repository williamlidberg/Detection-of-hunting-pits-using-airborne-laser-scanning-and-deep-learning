# adapted from https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html
import os
import torch
from torchvision.ops import masks_to_boxes
from torchvision.io import read_image
import tifffile
import argparse
from PIL import Image
import numpy as np

def boxes(temp_dir, labels_dir):
    for tile in os.listdir(labels_dir):
        if tile.endswith('.tif'):
            mask = tifffile.imread(labels_dir + tile)
            mask = mask.astype(np.uint8)
            mask_from_array = Image.fromarray(mask)
            
            # save image as png so it can be read with read_img. send help...
            temp_img = temp_dir + tile.replace('.tif', '.png')
            mask_from_array.save(temp_img)

            mask = read_image(temp_img)
            obj_ids = torch.unique(mask)
            # first id is the background, so remove it.
            obj_ids = obj_ids[1:] 
            masks = mask == obj_ids[:, None, None]
            boxes = masks_to_boxes(masks)
             # each box has a uniqe ID now but they should all be the same class
            #labels = torch.ones((masks.shape[0],), dtype=torch.int64) 
            labels = 1 # all bounding boxes are hunting pits.
            target = {}
            target["boxes"] = boxes
            target["labels"] = labels
            
            print(target) # How do we save the boundig boxes in a usefull format?

def clean_temp(temp_dir):
    for root, dir, fs in os.walk(temp_dir):
        for f in fs:
            os.remove(os.path.join(root, f))


def main(temp_dir, labels_dir):
    boxes(temp_dir, labels_dir)
    clean_temp(temp_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Extract topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('temp_dir', help= 'path to a temperary directory')
    parser.add_argument('labels_dir', help='Path to dem or folder of dems')
    #parser.add_argument('--boxes', help = 'bounding boxes') # can they be saved as COCO or PASCAL?

    args = vars(parser.parse_args())
    main(**args)