import os
import argparse
import tifffile

parser = argparse.ArgumentParser(description='normalize topogrpahical incides ')

def normalize_hillshade(hillshade, norm_img):
        img = tifffile.imread(hillshade)
        normed_shade = img/32767 # hillshade is a 16 bit signed integer file but only values between 0 and 32767 are used for hillshades
        tifffile.imwrite(norm_img, normed_shade.astype('float32'))

def normalize_slope(slope, norm_img):
        img = tifffile.imread(slope)
        normed_slope = img/90 # no slope can be flatter than 0 degrees or steeper than 90 degrees
        tifffile.imwrite(norm_img, normed_slope.astype('float32'))

def normalize_hpmf(hpmf, norm_img):
        img = tifffile.imread(hpmf)
        normed_hpmf = (img--1)/(2--1) 
        tifffile.imwrite(norm_img, normed_hpmf.astype('float32'))

def normalize_stdon(stdon, norm_img):
        img = tifffile.imread(stdon)
        normed_stdon = img/30 
        tifffile.imwrite(norm_img, normed_stdon.astype('float32'))


def main(input_hillshade, norm_hillshade, input_slope, norm_slope, input_hpmf, norm_hpmf, input_stdon, norm_stdon):

        for f in os.listdir(input_hillshade):
                if f.endswith('.tif'):
                        hillshade_img = input_hillshade + f
                        hillshade_norm = norm_hillshade + f
                        normalize_hillshade(hillshade_img, hillshade_norm)
                        
        for f in os.listdir(input_slope):
                if f.endswith('.tif'):
                        slope_img = input_slope + f
                        slope_norm = norm_slope + f
                        normalize_slope(slope_img, slope_norm)

        for f in os.listdir(input_hpmf):
                if f.endswith('.tif'):
                        hpmf_img = input_hpmf + f
                        hpmf_norm = norm_hpmf + f
                        normalize_hpmf(hpmf_img, hpmf_norm)

        for f in os.listdir(input_stdon):
                if f.endswith('.tif'):
                        stdon_img = input_stdon + f
                        stdon_norm = norm_stdon + f
                        normalize_stdon(stdon_img, stdon_norm)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='normalize topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_hillshade', help='Path to input hillshade')
    parser.add_argument('norm_hillshade', help = 'Path to normalized hillshade')
    parser.add_argument('input_slope', help='Path to input slope')
    parser.add_argument('norm_slope', help = 'Path to normalized slope')
    parser.add_argument('input_hpmf', help='Path to input high pass median filter')
    parser.add_argument('norm_hpmf', help = 'Path to normalized high pass medianfilter')
    parser.add_argument('input_stdon', help='Path to input stdon')
    parser.add_argument('norm_stdon', help = 'Path to normalized stdon')   
    args = vars(parser.parse_args())
    main(**args)