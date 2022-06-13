import os
import argparse
import tifffile
from osgeo import gdal
import utils.WriteGeotiff
import whitebox
wbt = whitebox.WhiteboxTools()

     #   # write image
     #   img_name = os.path.basename(bands[0]).split('.')[0]
     #   extent = gdal.Open(bands[0])
     #   utils.WriteGeotiff.write_gtiff(out, InutFileWithKnownExtent,
     #                                  os.path.join(out_path,
     #                                               '{}.{}'.format(img_name,
     #                                                              img_type)))


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

    def slope(self, input_path, normalized_slope, extent):
        wbt.slope(
        dem = input_path, 
        output = self.temp_dir + os.path.basename(input_path), 
        zfactor=None, 
        units= 'degrees')
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        normed_slope = img/90 # no slope can be flatter than 0 degrees or steeper than 90 degrees
        utils.WriteGeotiff.write_gtiff(normed_slope, extent, normalized_slope, gdal.GDT_Float32)

    def high_pass_median_filter(self, input_path, normalized_hpmf, extent):
        wbt.high_pass_median_filter(
        i = input_path, 
        output =  self.temp_dir + os.path.basename(input_path), 
        filterx=11, 
        filtery=11, 
        sig_digits=2)
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        normed_hpmf = (img--1)/(2--1)
        utils.WriteGeotiff.write_gtiff(normed_hpmf, extent, normalized_hpmf, gdal.GDT_Float32) 

    def spherical_std_dev_of_normals(self, input_path, normalized_stdon, extent):
        wbt.spherical_std_dev_of_normals(
        dem = input_path, 
        output = self.temp_dir + os.path.basename(input_path), 
        filter=5)
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        normed_stdon = img/30
        utils.WriteGeotiff.write_gtiff(normed_stdon, extent, normalized_stdon, gdal.GDT_Float32)  

def clean_temp(temp_dir):
    for root, dir, fs in os.walk(temp_dir):
        for f in fs:
            os.remove(os.path.join(root, f))


def main(temp_dir, input_path, output_path_hillshade, output_path_slope, output_path_hpmf, output_path_stdon):
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
        slope = os.path.join(output_path_slope,'{}.{}'.format(img_name, 'tif'))
        high_pass_median_filter = os.path.join(output_path_hpmf,'{}.{}'.format(img_name, 'tif'))
        spherical_std_dev_of_normals = os.path.join(output_path_stdon,'{}.{}'.format(img_name, 'tif'))
        
        extent = gdal.Open(img_path) # extract projection and extent from input image
        topographical = Topographical_indices(temp_dir)
        clean_temp(temp_dir)
        topographical.hillshade(img_path, hillshade, extent)
        clean_temp(temp_dir)
        topographical.slope(img_path, slope, extent)
        clean_temp(temp_dir)
        topographical.high_pass_median_filter(img_path, high_pass_median_filter, extent)
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
    parser.add_argument('output_path_slope', help = 'directory to store slope images')
    parser.add_argument('output_path_hpmf', help = 'directory to store hpmf images')
    parser.add_argument('output_path_stdon', help='directory to store stdon images')
    args = vars(parser.parse_args())
    main(**args)