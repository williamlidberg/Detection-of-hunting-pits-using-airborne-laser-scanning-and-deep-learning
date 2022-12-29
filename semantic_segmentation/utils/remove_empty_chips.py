import os
import arcpy
import numpy as np
import argparse

# parser = argparse.ArgumentParser(description='remove image chips without enough labeled pixels ')

#def main(imagedirectory, labeldirectory, numpixels):
def remove_tiles(imagedirectory, labeldirectory, numpixels):
    for tile in os.listdir(labeldirectory):
        if tile.endswith('.tif'):
            # input data
            imagewithpath = imagedirectory + tile
            labelwithpath = labeldirectory + tile

            npArray = arcpy.RasterToNumPyArray(labelwithpath)
            npArray1 = np.ma.masked_array(npArray, np.isnan(npArray))
            tilesum = np.sum(npArray)
            if tilesum < numpixels:
                #ListofTilesWithoutDitches.append(labelwithpath)
                arcpy.Delete_management(imagewithpath)
                arcpy.Delete_management(labelwithpath)
                print('removed', tile)

# if __name__ == '__main__':
#     import argparse

#     parser = argparse.ArgumentParser(
#                        description='Extract topographical indicies '
#                                    'image(s)',
#                        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
#     parser.add_argument('imagedirectory', help='Path directory of image chips')
#     parser.add_argument('labeldirectory', help = 'path to directory of label chips')
#     parser.add_argument('numpixels', help = 'threshold of minimum number of pixels to keep', type=int)

    
#     args = vars(parser.parse_args())
#     main(**args)
images = 'Y:/William/Kolbottnar/data/hillshade/images/'
labels = 'Y:/William/Kolbottnar/data/hillshade/labels/'
min_number_pixels = 1

remove_tiles(images, labels, min_number_pixels)
