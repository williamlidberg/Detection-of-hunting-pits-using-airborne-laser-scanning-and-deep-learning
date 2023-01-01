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

    def spherical_std_dev_of_normals(self, input_path, normalized_stdon, extent):
        wbt.spherical_std_dev_of_normals(
        dem = input_path, 
        output = self.temp_dir + os.path.basename(input_path), 
        filter=11)
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        normed_stdon = img/30
        utils.WriteGeotiff.write_gtiff(normed_stdon, extent, normalized_stdon, gdal.GDT_Float32)  

    def maxelevationdeviation(self, input_path, normalized_MED, extent):
        wbt.max_elevation_deviation(
            dem = input_path, 
            out_mag = self.temp_dir + os.path.basename(input_path), 
            out_scale= self.temp_dir + os.path.basename(input_path.replace('.tif', '_scale.tif')), 
            min_scale=0, 
            max_scale=10, 
            step=1)
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        img[img > 10] = 10 # we are not interested in pits deeper than 2 m.
        img[img < -10] = -10
        normed_MED = (img--10)/(10--10)
        utils.WriteGeotiff.write_gtiff(normed_MED, extent, normalized_MED, gdal.GDT_Float32)

    def multiscaleelevationpercentile(self, input_path, normalized_MSEP, extent):
        wbt.multiscale_elevation_percentile(
            dem = input_path, 
            out_mag = self.temp_dir + os.path.basename(input_path), 
            out_scale= self.temp_dir + os.path.basename(input_path.replace('.tif', '_scale.tif')), 
            sig_digits=3, 
            min_scale=5, 
            step=1, 
            num_steps=10, 
            step_nonlinearity=1.0)
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        normed_MSEP = img/100 # devide by 100 to change scale to 0-1.
        utils.WriteGeotiff.write_gtiff(normed_MSEP, extent, normalized_MSEP, gdal.GDT_Float32)

    def depthinsink(self, input_path, normalized_depthinsink, extent):
        wbt.depth_in_sink(
            dem = input_path, 
            output = self.temp_dir + os.path.basename(input_path), 
            zero_background=True)
        img = tifffile.imread(self.temp_dir + os.path.basename(input_path))
        img[img > 10] = 10 # we are not interested in pits deeper than 10 m anyway
        normed_depthinsink = img/10 # devide by 10 to change scale to 0-1.
        utils.WriteGeotiff.write_gtiff(normed_depthinsink, extent, normalized_depthinsink, gdal.GDT_Float32)

def clean_temp(temp_dir):
    for root, dir, fs in os.walk(temp_dir):
        for f in fs:
            os.remove(os.path.join(root, f))


def main(temp_dir, input_path, output_path_hillshade, output_path_elevation_above_pit, output_path_minimal_curvature,output_path_profile_curvature, output_path_stdon, output_maxelevationdeviation, output_multiscaleelevationpercentile, output_depthinsink):
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
        maxelevationdeviation = os.path.join(output_maxelevationdeviation,'{}.{}'.format(img_name, 'tif')) 
        multiscaleelevationpercentile = os.path.join(output_multiscaleelevationpercentile,'{}.{}'.format(img_name, 'tif')) 
        depthinsink = os.path.join(output_depthinsink,'{}.{}'.format(img_name, 'tif')) 

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
        topographical.maxelevationdeviation(img_path, maxelevationdeviation, extent)
        clean_temp(temp_dir)
        topographical.multiscaleelevationpercentile(img_path, multiscaleelevationpercentile, extent)
        clean_temp(temp_dir)
        topographical.depthinsink(img_path, depthinsink, extent)
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
    parser.add_argument('output_maxelevationdeviation', help = 'directory to store output_maxelevationdeviation images')
    parser.add_argument('output_multiscaleelevationpercentile', help = 'directory to store output_multiscaleelevationpercentileimages')
    parser.add_argument('output_depthinsink', help='directory to store output_depthinsink images')
    args = vars(parser.parse_args())
    main(**args)

