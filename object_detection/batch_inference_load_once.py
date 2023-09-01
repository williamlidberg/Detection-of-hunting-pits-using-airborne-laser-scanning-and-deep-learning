# This works best on CPU.
import sys
sys.path.append("./yolor")
sys.path.append("./pipeline_utils")

import read_tiff_data

import pdb
import torch
import requests
import urllib.request
import argparse
import os
import detect_tile_gpkg_1_dim_load_once as detect_tile_gpkg_1_dim_load_once
import errno
from tqdm.auto import tqdm
from utils.torch_utils import select_device, time_synchronized
import pathlib
import torch.backends.cudnn as cudnn
#from models.models import *
from YolorModels import Darknet
import time
import shutil
from tifffile import imread
import platform
import glob
import math

#suppressed all warning messages. Be careful to use this
import warnings
warnings.filterwarnings("ignore")
    
    
pltf = platform.system()
if pltf == 'Linux':
    print("on Linux system")
    pathlib.WindowsPath = pathlib.PosixPath
else:
    print("on Windows system")
    pathlib.PosixPath = pathlib.WindowsPath

__here__ = os.path.dirname(__file__)
# Add the directory of train.py to the python path to be able to laod modules. Needed after the latest AML runtime update.
sys.path.insert(1, './' + __here__)

def ensure_dir(directory):
    try:
        os.makedirs(directory)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise
   

def main(opt):
    image_service, out, source, weights, view_img, save_txt, imgsz, cfg, names, filename, outfile = \
        opt.image_service, opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names, opt.filename, opt.outfile

    # Initialize
    device = select_device(opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder
    os.makedirs(out)  # make new output folder
    print("device: {}".format(device))

    #cfg = os.path.join(__here__,cfg)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    print("current location: {}, cfg location: {}, weights location: {}".format(__here__, cfg, weights))

    # Load model
    model = Darknet(cfg, imgsz).cuda()
    model.load_state_dict(torch.load(opt.weights[0], map_location=device)['model'])
    
    model.to(device).eval()
    if half:
        model.half()  # to FP16

    # Run inference
#    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img

    img = torch.zeros((1, 1, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once
    cudnn.benchmark = False
    
    start = time_synchronized()
    current_time = 0
    
    # Hitta alla TIFF-bilder i katalogen
    tif_files = glob.glob(os.path.join(opt.source, '*.tif'))

    # Loopa igenom alla TIFF-bilder
    for tif_file in tif_files:
        tile_start_time = time_synchronized()
        tiff_file = open(tif_file, 'rb')
        tiff_tags = read_tiff_data.get_tiff_data(tiff_file)    

        meter_per_pixel = read_tiff_data.get_meter_per_pixel(tiff_tags)
        gps_coordinate = read_tiff_data.get_gps_coordinates(tiff_tags)

        print(f'Filename {tif_file}, meter_per_pixel {meter_per_pixel}, gps_coord {gps_coordinate}')

        opt.filename = os.path.basename(tif_file)

        detect_tile_gpkg_1_dim_load_once.detect(model, device, half, opt, local_pt_data_path=opt.source, local_results_path=opt.output, batch_size=opt.batch_size)
        
        tile_run_time = time_synchronized() - tile_start_time
        print("tile run: {}".format(tile_run_time))
        current_time += tile_run_time
                
        days, hours, minutes, seconds = convert_seconds_to_readable_format(current_time)
        print("current time spent:  days: {}, hours: {}, minutes: {}, seconds: {}".format(days, hours, minutes, seconds))
            
      

def convert_seconds_to_readable_format(seconds):
    split_days = math.floor(seconds/3600/24)
    split_hours = math.floor(seconds/3600) - split_days * 24
    split_minutes = math.floor(seconds/60.0) - split_hours*60 - split_days * 24 * 60
    split_seconds = seconds % 60
    return split_days, split_hours, split_minutes, split_seconds
            
if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--coordinates', action='store', dest='param_coordinates',
        help='ex: 587500,6563500,588500,6565500')
        
    parser.add_argument('--year', action='store', dest='param_year', default='2021',
        help='ex: 2019')    
    
    parser.add_argument('--offset', action='store', dest='param_offset', default='1000',
        help='the offset for each tif image, ex 1000 (Meter)')
        
    parser.add_argument('--rendering_rule' , action='store', dest='param_rendering_rule', default='KraftigareFargmattnad',
        help='the redering rule')
        
    parser.add_argument('--input_path', action='store', dest='param_input_datapath_name', default='input',
        help='Path to the input data')
    parser.add_argument('--output_path', action='store', dest='param_output_datapath_name', default='output',
        help='Path to the out data in the bloc container')
    parser.add_argument('--batch-size', type=int, default=8, help='batch images', action='store', dest='batch_size')

    parser.add_argument('--weights', nargs='+', type=str, default='yolor_best_overall.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6_kolbottnar.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='../data.names', help='*.cfg path')
    parser.add_argument('--filename', type=str, default='input.tif', help='*.tif')
    parser.add_argument('--outfile', type=str, default='input.gpkg', help='*.gpkg')
    parser.add_argument('--image_service', type=str, default='ortophoto_2_0', help='ortophoto_2_0, hojdmodell')      
        
    opt = parser.parse_args()
    print(opt)
    main(opt)
    # Move results to output
    sourcefile =  os.path.join("inference",opt.outfile)
    destfile =  os.path.join(opt.output,opt.outfile)
    shutil.move(sourcefile, destfile)
    print("Successfully moved {} to {}".format(sourcefile, destfile))

    