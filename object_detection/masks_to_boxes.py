# adapted from https://pytorch.org/vision/main/auto_examples/plot_repurposing_annotations.html
import os
import torch
from torchvision.ops import masks_to_boxes
from torchvision.io import read_image
import tifffile
import argparse
from PIL import Image
import numpy as np
import pybboxes as pbx


def boxes(temp_dir, labels_dir, image_size, label_class, bounding_box_dir):
    for tile in os.listdir(labels_dir):
        if tile.endswith('.tif'):
            mask = tifffile.imread(labels_dir + tile)
            mask = mask.astype(np.uint8)
            mask_from_array = Image.fromarray(mask)
            temp_img = temp_dir + tile.replace('.tif', '.png')
            mask_from_array.save(temp_img) # save image as png so it can be read with read_img. There must be a better way to do this.
            mask = read_image(temp_img) 
            obj_ids = torch.unique(mask)
            obj_ids = obj_ids[1:] 
            masks = mask == obj_ids[:, None, None]
            boxes = masks_to_boxes(masks)
            
            # convert voc style of bounding boxes to yolo style.
            boxers = torch.Tensor(boxes).numpy()
            W = image_size
            H = image_size
            for box in boxers: 
                x_min = box.item(0)
                y_min = box.item(1)
                x_max = box.item(2)
                y_max = box.item(3)
                voc = x_min, y_min, x_max, y_max
                yolo = pbx.convert_bbox(voc, from_type="voc", to_type="yolo", image_width=W , image_height=H)
                yolo_list = [str(i) for i in list(yolo)]
                yolo_list.insert(0,str(label_class))
                with open(os.path.join(bounding_box_dir, tile.replace('.tif', '.txt')), 'a') as f:
                    f.write(" ".join(yolo_list))
                    f.write('\n')

def clean_temp(temp_dir):
    for root, dir, fs in os.walk(temp_dir):
        for f in fs:
            os.remove(os.path.join(root, f))


def main(temp_dir, labels_dir, image_size, label_class, bounding_box_dir):
    boxes(temp_dir, labels_dir, image_size, label_class, bounding_box_dir)

    clean_temp(temp_dir)
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Extract topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('temp_dir', help= 'Path to temp dir')
    parser.add_argument('labels_dir', help= 'Path to segmentation masks or folder of dems')
    parser.add_argument('image_size', type=int, help= 'size of image in number of pixels')
    parser.add_argument('label_class', type=int, help= 'class to give the labels')
    parser.add_argument('bounding_box_dir', help= 'Path to dem or folder of dems')
    args = vars(parser.parse_args())
    main(**args)
