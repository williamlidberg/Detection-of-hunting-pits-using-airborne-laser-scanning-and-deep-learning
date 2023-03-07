import os
from tifffile import tifffile
import numpy as np
import shutil
from pathlib import Path

def main(original_dir, output_dir, numpixels):
    for subdir in os.listdir(original_dir):
        if subdir != 'backup': #temporary fix
            print('making new directories for output chips')
            os.makedirs((output_dir + subdir))
            if subdir == 'labels':
                print('copying labaled chips to new directory')
                for chip in os.listdir((original_dir + subdir)):
                    in_name = (original_dir + subdir + '/' + chip)
                    out_name = (output_dir + subdir + '/' + chip)
                    image = tifffile.imread(in_name)
                    tilesum = np.sum(image)
                    if tilesum > numpixels:
                        shutil.copy(in_name, out_name)
    list_of_labels =[]
    for label_chip in os.listdir((output_dir + '/labels/')):
        path = Path((output_dir + '/labels/' + label_chip))
        list_of_labels.append((Path(path)).stem)

    for subdir in os.listdir(original_dir):
        print('copying', len(list_of_labels), 'files from ', subdir)
        if subdir != 'backup':
            if subdir != 'labels':
                for chip in os.listdir((original_dir + subdir + '/')):
                    inpath = Path((original_dir + subdir + '/'+ chip))
                    outpath = (output_dir + subdir + '/'+ chip)
                    chipname = inpath.stem
                    if chipname in list_of_labels:
                        shutil.copy(inpath, outpath)


if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='copy chips with labeled pixels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('original_dir', help='path to label chips')
    parser.add_argument('output_dir', help='path to label chips')
    parser.add_argument('numpixels', help = 'minimum number of pixels to copy files', type=int)



    args = vars(parser.parse_args())
    main(**args)