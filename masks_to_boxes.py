# adapted from https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html
import os
import torch
from torchvision.ops import masks_to_boxes
from torchvision.io import read_image
import tifffile
import argparse
from PIL import Image
import numpy as np

def boxes(temp_dir, labels_dir, bounding_box_dir):
    for tile in os.listdir(labels_dir):
        if tile.endswith('.tif'):
            mask = tifffile.imread(labels_dir + tile)
            mask = mask.astype(np.uint8)
            mask_from_array = Image.fromarray(mask)
            
            
            temp_img = temp_dir + tile.replace('.tif', '.png')
            mask_from_array.save(temp_img) # save image as png so it can be read with read_img. There must be a better way to do this.
            mask = read_image(temp_img) 
            obj_ids = torch.unique(mask)
            obj_ids = obj_ids[1:] # first id is the background, so remove it.
            masks = mask == obj_ids[:, None, None]
            
            boxes = masks_to_boxes(masks)
            print(boxes)
            np.savetxt(os.path.join(bounding_box_dir, tile.replace('.tif', '.txt')), torch.Tensor(boxes).numpy())
            #x_min = boxes[0]
            #y_min = boxes[1]
            #x_max = boxes[2]
            #y_max = boxes[3] 
            #print(x_min) 
         #   with open(os.path.join(bounding_box_dir, tile.replace('.tif', '.txt')), 'w') as f:
         #       for feature in boxes:
         #           f.write(feature)
         #           f.write('\n')


def clean_temp(temp_dir):
    for root, dir, fs in os.walk(temp_dir):
        for f in fs:
            os.remove(os.path.join(root, f))


def main(temp_dir, labels_dir, bounding_box_dir):
    boxes(temp_dir, labels_dir, bounding_box_dir)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Extract topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('temp_dir', help= 'Path to temp dir')
    parser.add_argument('labels_dir', help= 'Path to segmentation masks or folder of dems')
    parser.add_argument('bounding_box_dir', help= 'Path to dem or folder of dems')
    #parser.add_argument('--size',type=int, help= 'image size')
    #parser.add_argument('--boxes', help = 'bounding boxes') # can they be saved as COCO or PASCAL?

    args = vars(parser.parse_args())
    main(**args)