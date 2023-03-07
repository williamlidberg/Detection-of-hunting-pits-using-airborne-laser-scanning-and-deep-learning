import os
import matplotlib.pyplot as plt
import tifffile
import argparse

parser = argparse.ArgumentParser(description='normalize topogrpahical incides ')



def main(image, plot):
    for f in os.listdir(image):
        if f.endswith('.tif'):
            in_image = image + f
            out_image = plot + f.replace('.tif','.png')
            im = tifffile.imread(in_image)
            # calculate mean value from RGB channels and flatten to 1D array
            vals = im.mean(axis=1).flatten()
            # plot histogram with 100 bins
            b, bins, patches = plt.hist(vals, 100)
            plt.xlim([0,32767])
            plt.savefig(out_image)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='normalize topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('image', help='Path to input hillshade')
    parser.add_argument('plot', help = 'Path to normalized hillshade')

    args = vars(parser.parse_args())
    main(**args)
