import os
import shutil

def main(segmentation_masks, detection_masks, selected_detection_masks):
    for chip in os.listdir(segmentation_masks):
        if chip.endswith('.tif'):
            input_chip = detection_masks + chip
            output_chip = selected_detection_masks + chip
            shutil.copy(input_chip, output_chip)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Merge bounding boxes '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('segmentation_masks', help= 'path to directory with segmentation masks')
    parser.add_argument('detection_masks', help= 'path to splited detection masks')
    parser.add_argument('selected_detection_masks', help= 'path do directory where selected detection masks will be saved')

    args = vars(parser.parse_args())
    main(**args)