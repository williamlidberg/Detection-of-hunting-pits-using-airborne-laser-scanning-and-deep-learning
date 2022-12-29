import os
from splitraster import geo

def main(input_path, output_path, tile_size):

    # setup paths
    if not os.path.exists(input_path):
        raise ValueError('Input path does not exist: {}'.format(input_path))
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.endswith('.tif')]

    else:
        imgs = [input_path]

    imgs.sort()
    for input_image_path in imgs:
        n = geo.split_image(input_image_path, output_path, tile_size, repetition_rate=0, overwrite=False)
        print(f"{n} tiles sample of {input_image_path} are added at {output_path}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='split input raster tiles into smaller chips '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='path to input images or directory of images')
    parser.add_argument('output_path', help = 'path to directory to store hillshade images')
    parser.add_argument('--tile_size',help = 'size of image chips. Default is 500 pixels', default=500, type=int)
    args = vars(parser.parse_args())
    main(**args)