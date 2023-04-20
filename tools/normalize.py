import os
import argparse
import tifffile

def main(input_path, output_path):
    if not os.path.exists(input_path):
        raise ValueError('Input path does not exist: {}'.format(input_path))
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.endswith('.tif')]
    else:
        imgs = [input_path]
    
    for img_path in imgs:
        original = tifffile.imread(img_path)
        normalised = original * 255
        tifffile.imwrite(output_path + os.path.basename(img_path), normalised)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
                       description='normalises data from 0-1 to 0-255 for YOLO',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='path to input images or directory of images')
    parser.add_argument('output_path', help = 'path to directory to store new images')

    args = vars(parser.parse_args())
    main(**args)
