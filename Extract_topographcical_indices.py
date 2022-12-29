import os
import argparse
import tifffile
from osgeo import gdal
import utils.WriteGeotiff
import whitebox
wbt = whitebox.WhiteboxTools()


class Topographical_indices:
    def __init__(self, temp_dir):
        self.temp_dir = temp_dir

    def hillshade(self, input_path, normalized_hillshade, extent):
        wbt.multidirectional_hillshade(
        dem = input_path, 
        output = self.temp_dir + os.path.basename(input_path), 
        altitude=45.0, 
        zfactor=None, 
        full_mode=False)
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        normed_shade = img/32767 # hillshade is a 16 bit signed integer file but only values between 0 and 32767 are used for hillshades
        utils.WriteGeotiff.write_gtiff(normed_shade, extent, normalized_hillshade, gdal.GDT_Float32)

    def elevation_above_pit(self, input_path, normalized_elevation_above_pit, extent):
        wbt.elev_above_pit(
            dem = input_path, 
            output = self.temp_dir + os.path.basename(input_path))
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        img[img > 2] = 2 # we are not interested in pits deeper than 2 m.
        normed_elevation_above_pit = img/2
        utils.WriteGeotiff.write_gtiff(normed_elevation_above_pit, extent, normalized_elevation_above_pit, gdal.GDT_Float32)

    def minimal_curvature(self, input_path, normalized_minimal_curvature, extent):
        wbt.minimal_curvature(
            dem = input_path, 
            output = self.temp_dir + os.path.basename(input_path), 
            log=True, 
            zfactor=None)
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        img[img > 5] = 5 # we are not interested in pits deeper than 2 m.
        img[img < -5] = -5
        normed_minimal_curvature = (img--5)/(5--5)
        utils.WriteGeotiff.write_gtiff(normed_minimal_curvature, extent, normalized_minimal_curvature, gdal.GDT_Float32)

    def profile_curvature(self, input_path, normalized_profile_curvature, extent):
        wbt.profile_curvature(
            dem = input_path, 
            output = self.temp_dir + os.path.basename(input_path), 
            log=False, 
            zfactor=None)
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        img[img > 1] = 1 # we are not interested in pits deeper than 2 m.
        img[img < -2] = -2
        normed_profile_curvature = (img--2)/(1--2)
        utils.WriteGeotiff.write_gtiff(normed_profile_curvature, extent, normalized_profile_curvature, gdal.GDT_Float32)

    # def high_pass_median_filter(self, input_path, normalized_hpmf, extent):
    #     wbt.high_pass_median_filter(
    #     i = input_path, 
    #     output =  self.temp_dir + os.path.basename(input_path), 
    #     filterx=11, 
    #     filtery=11, 
    #     sig_digits=2)
    #     img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
    #     normed_hpmf = (img--1)/(2--1)
    #     utils.WriteGeotiff.write_gtiff(normed_hpmf, extent, normalized_hpmf, gdal.GDT_Float32) 

    def spherical_std_dev_of_normals(self, input_path, normalized_stdon, extent):
        wbt.spherical_std_dev_of_normals(
        dem = input_path, 
        output = self.temp_dir + os.path.basename(input_path), 
        filter=11)
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        normed_stdon = img/30
        utils.WriteGeotiff.write_gtiff(normed_stdon, extent, normalized_stdon, gdal.GDT_Float32)  

def clean_temp(temp_dir):
    for root, dir, fs in os.walk(temp_dir):
        for f in fs:
            os.remove(os.path.join(root, f))


def main(temp_dir, input_path, output_path_hillshade, output_path_elevation_above_pit, output_path_minimal_curvature,output_path_profile_curvature, output_path_stdon):
#    setup paths
    if not os.path.exists(input_path):
        raise ValueError('Input path does not exist: {}'.format(input_path))
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.endswith('.tif')]
    else:
        imgs = [input_path] 
    for img_path in imgs:
        img_name = os.path.basename(img_path).split('.')[0]
        
        # outputs 
        hillshade = os.path.join(output_path_hillshade,'{}.{}'.format(img_name, 'tif'))
        elevation_above_pit = os.path.join(output_path_elevation_above_pit,'{}.{}'.format(img_name, 'tif'))
        minimal_curvature = os.path.join(output_path_minimal_curvature,'{}.{}'.format(img_name, 'tif'))
        profile_curvature = os.path.join(output_path_profile_curvature,'{}.{}'.format(img_name, 'tif'))
        spherical_std_dev_of_normals = os.path.join(output_path_stdon,'{}.{}'.format(img_name, 'tif'))
        
        extent = gdal.Open(img_path) # extract projection and extent from input image
        topographical = Topographical_indices(temp_dir)
        clean_temp(temp_dir)
        topographical.hillshade(img_path, hillshade, extent)
        clean_temp(temp_dir)
        topographical.elevation_above_pit(img_path, elevation_above_pit, extent)
        clean_temp(temp_dir)
        topographical.minimal_curvature(img_path, minimal_curvature, extent)
        clean_temp(temp_dir)
        topographical.profile_curvature(img_path, profile_curvature, extent)
        clean_temp(temp_dir)
        topographical.spherical_std_dev_of_normals(img_path, spherical_std_dev_of_normals, extent)
        clean_temp(temp_dir)

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Extract topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('temp_dir', help= 'path to a temperary directory')
    parser.add_argument('input_path', help='Path to dem or folder of dems')
    parser.add_argument('output_path_hillshade', help = 'directory to store hillshade images')
    parser.add_argument('output_path_elevation_above_pit', help = 'directory to store elevation_above_pitimages')
    parser.add_argument('output_path_minimal_curvature', help = 'directory to store output_path_minimal_curvature images')
    parser.add_argument('output_path_profile_curvature', help = 'directory to store hpmf images')
    parser.add_argument('output_path_stdon', help='directory to store stdon images')
    args = vars(parser.parse_args())
    main(**args)