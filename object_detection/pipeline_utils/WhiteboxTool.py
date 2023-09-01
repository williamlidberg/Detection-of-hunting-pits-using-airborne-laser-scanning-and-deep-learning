import ssl
import whitebox
import numpy as np
from tifffile import imread


def init_whitebox():
    '''initialize the whitebox tool. The instance sets as a global variable.
      Since it's optional to use whitebox, it would be unflexible to pass the instance around. 
      We also only want to load it once'''
    ssl._create_default_https_context = ssl._create_unverified_context
    global wbt
    wbt = whitebox.WhiteboxTools()
    wbt.work_dir = '/mnt'

# def hillshade(input_path, output_path, wbt):
#     wbt.multidirectional_hillshade(input_path, output_path)
#     wbt.multidirectional_hillshade(elevation: numpy.ndarray, azimuth: float, altitude: float, output: numpy.ndarray)
#     return imread(output_path)

def hillshade(input_array: tuple) -> tuple:
    #wbt.multidirectional_hillshade(input_path, output_path)
    output_arr = np.array([],dtype='float32')
    global wbt
    wbt.multidirectional_hillshade(input_array, output_arr)
    return output_arr

# def high_pass_median_filter(input_path, output_path, wbt):
#     wbt.high_pass_median_filter(input_path, output_path)
#     return imread(output_path)

def high_pass_median_filter(input_array: tuple) -> tuple:
    output_arr = np.array([])
    global wbt
    wbt.high_pass_median_filter(input_array, output_arr)
    return imread(output_arr)


def local_slope(input_path: str, output_path: str, wbt) -> tuple:
    wbt.standard_deviation_of_slope(input_path, output_path,  filterx=3, filtery=3)
    return imread(output_path)


def normalize(img_arr: tuple, std: int, median: int) -> tuple:
    img_arr = img_arr - (median - 2*std)
    img_arr = img_arr / (4*std)
    img_arr *= 255
    img_arr = np.clip(img_arr, 0, 255)
    img_arr = img_arr.astype(np.uint8)
    return img_arr


def normalize_hill_shade(img_arr: tuple) -> tuple:
    #img_arr = imread(file_path)
    std = 1456
    median = 23114
    #img_arr = normalize(img_arr,std,median)
    
    #hill shade 32bit signed
    img_arr = (img_arr/32767)*255
    img_arr = img_arr.astype(np.uint8)
   # imwrite(file_path.replace('.tif','_normalized.tif'),img_arr)
    return img_arr


def normalize_local_slope(file_path: str) -> tuple:
    img_arr = imread(file_path)
    std = 1.234
    median = 1.767
    img_arr = normalize(img_arr, std, median)
    #img_arr = (img_arr/90)*255
    #img_arr = img_arr.astype(np.uint8)
   # imwrite(file_path.replace('.tif','_normalized.tif'),img_arr)
    return img_arr


def normalize_high_pass_median_filter(img_arr: tuple) -> tuple:
    #img_arr = imread(file_path)
    std = 0.1009521335363388
    median = 0
    img_arr = normalize(img_arr, std, median)
    #img_arr = img_arr.astype(np.uint8)
   # imwrite(file_path.replace('.tif','_normalized.tif'),img_arr)
    return img_arr