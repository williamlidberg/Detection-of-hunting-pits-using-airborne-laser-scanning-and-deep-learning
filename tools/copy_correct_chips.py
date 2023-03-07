import os
import shutil


def main(original_dir, box_dir, final_dir):
    """
    Loops over subdirs and copy chips that have valid bounding boxes to 
    new directory. 
    """
    valid_boxes = os.listdir(box_dir)
    for box in valid_boxes:
        inbox = box_dir + box
        outbox = final_dir + '/bounding_boxes/' + box
        shutil.copy(inbox, outbox)
    print('copied bounding boxes')

    for subdir, dirs, chips in os.walk(original_dir):
        for chip in chips:
            chipname = chip.replace('.tif','.txt')
            if chipname in valid_boxes:
                inchip = os.path.join(subdir, chip)
                outchip = final_dir + '/' + (os.path.basename(subdir))+ '/' + chip
                shutil.copy(inchip, outchip )
    print('copied image chips')


if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='copy chips with labeled pixels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('original_dir', help='path to label chips')
    parser.add_argument('box_dir', help = 'path to correct_bounding boxes')
    parser.add_argument('final_dir', help = 'path to correct_bounding boxes')


    args = vars(parser.parse_args())
    main(**args)