import os
import argparse
import whitebox
import normalize_indices
wbt = whitebox.WhiteboxTools()

parser = argparse.ArgumentParser(description='Extract topogrpahical incides from DEMs. ')

def main(input_path, output_path_hillshade, output_path_slope, output_path_hpmf):

    # setup paths
    if not os.path.exists(input_path):
        raise ValueError('Input path does not exist: {}'.format(input_path))
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.endswith('.tif')]

    else:
        imgs = [input_path]

    
    for img_path in imgs:
        predicted = []
        print(img_path)
        img_name = os.path.basename(img_path).split('.')[0]
        
        hillshade =  os.path.join(output_path_hillshade,'{}.{}'.format(img_name, 'tif'))
        slope = os.path.join(output_path_slope,'{}.{}'.format(img_name, 'tif'))
        high_pass_median_filter = os.path.join(output_path_hpmf,'{}.{}'.format(img_name, 'tif'))

        wbt.multidirectional_hillshade(
            dem = img_path, 
            output = hillshade, 
            altitude=45.0, 
            zfactor=None, 
            full_mode=False
        )

        wbt.slope(
            dem = img_path, 
            output = slope, 
            zfactor=None, 
            units= 'degrees'
        )

        wbt.high_pass_median_filter(
            i = img_path, 
            output = high_pass_median_filter, 
            filterx=11, 
            filtery=11, 
            sig_digits=2
        )


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Extract topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to dem or folder of dems')
    parser.add_argument('output_path_hillshade', help = 'directory to store hillshade images')
    parser.add_argument('output_path_slope', help = 'directory to store slope images')
    parser.add_argument('output_path_hpmf', help = 'directory to store hpmf images')
    
    args = vars(parser.parse_args())
    main(**args)