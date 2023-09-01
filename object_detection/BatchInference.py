import argparse
import time
from tifffile import imread
import torch
from numpy import random
import sys
import os
import numpy as np
sys.path.append("pipeline_utils")
import ImageRequest as ir
import ImageProcessing as ip
import GeoPackage as gp
import WhiteboxTool as wb
import Common as co
import YoloConverter as yc
import logging

logging.basicConfig(level='INFO')
log = logging.getLogger()
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import pdb
import shutil

def get_coordinates_from_string(coordinates_string: str) -> tuple:
    '''Takes a string of coordinates. ex. "510000,6440000,515000,6445000".
    Parse it to seperate integers'''
    coordinates = coordinates_string.split(',')
    minY, minX, maxY, maxX = [int(coord) for coord in coordinates]
    return minY, minX, maxY, maxX


def get_combined_image(coordinates: list, json_configurations: dict, configuration: str, img_size: int):
    '''Takes a list of coordinates that is the definition of the image rectangle. 
    A dict that holds all configurations configuration
    The chosen configuration key for the dict
    The size of the image to be downloaded'''
    channels = []
    #iterate throgh all APIS
    for i, api in enumerate(json_configurations[configuration]['apis']):
        image_path = ir.get_image(coordinates, img_size, api, 'input', 'image')
        channels.append(image_path)
        combined_image_path = 'input/combined_{}.tiff'.format(coordinates)
    combined_image_path, number_of_channels = ip.combine_image_channels(channels, combined_image_path)
    return combined_image_path, channels


def detect():
    '''This is the main method for the inference script. Arguments must be set with argparse when calling the script.
    The method is to iterate a rectangle over the defined coordinates until all area is covered.
    In each iteration, new images will be downloaded based on the chosen configuration.
    For each image, the loaded AI-model will using object detection.
    Every detection that has a confidence higher than the confidence threshold will be stored into a .gpkg file as boxes and points'''
    weights, save_txt, img_size, configuration, coordinates, offset, geographical_image_size, save_image, build_dataset_path, algorithm, output = opt.weights, opt.save_txt, opt.img_size, opt.configuration, opt.coordinates, opt.offset, opt.geographical_image_size, opt.save_image, opt.build_dataset_path, opt.algorithm, opt.output
    

    if algorithm == 'yolov7':
        log.info("load modules for yolov7")
        sys.path.append("./yolov7")
        from models.experimental import attempt_load
        from utils.general import  non_max_suppression, xyxy2xywh
        from Plots import plot_one_box
        from utils.torch_utils import select_device, time_synchronized
    elif algorithm == 'yolor':
        log.info("load modules for yolovr")
        sys.path.append("./yolor")
        from utils.torch_utils import select_device, time_synchronized
        from utils.general import ( non_max_suppression, xyxy2xywh)
        from PlotsYolor import plot_one_box
        from YolorModels import Darknet
        from YolorModels import parse_model_cfg
    json_configuration = co.get_configuration(configuration)
    
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # # Load model
    if algorithm.lower() == 'yolov7':
        log.info("load model for yolov7")
        model = attempt_load(weights, map_location=device)  # load FP32 model
        number_of_in_channels = model.model[0].conv.in_channels
        names = model.module.names if hasattr(model, 'module') else model.names
        
    elif algorithm.lower() == 'yolor':
        log.info("load model for yolor")
        names = opt.names
        model = Darknet(opt.yolor_cfg, (img_size,img_size))
        number_of_in_channels = parse_model_cfg(opt.yolor_cfg)[0]['channels']
        model.load_state_dict(torch.load(weights[0], map_location=device)['model'])
        model.to(device).eval()
    if half:
        model.half()  # to FP16

 
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    log.info("colors for classes: {}".format(colors))
    if opt.init_whitebox:
        wb.init_whitebox()

    if build_dataset_path:
        co.ensure_dir(os.path.join(build_dataset_path, 'images'))
        co.ensure_dir(os.path.join(build_dataset_path, 'labels'))
    co.ensure_dir(output)
    minY, minX, maxY, maxX = get_coordinates_from_string(coordinates)


    t0 = time.time()
    for y in range(minY, maxY, geographical_image_size + offset):
        for x in range(minX, maxX, geographical_image_size + offset):
            points = []
            bounding_boxes = []
            combined_image_path, channels = get_combined_image([y,x,y+geographical_image_size,x+geographical_image_size], json_configuration, configuration, img_size)
            meter_per_pixel, image_gps_coordinates = ip.read_tiff_data(combined_image_path, ['meter_per_pixel', 'gps_coordinates'])
            img = imread(combined_image_path)

            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            # jpg model that using single channel data must have converted images to work
            if number_of_in_channels == 3 and len(img.shape) == 2:
                img = img.unsqueeze(0).repeat(3,1,1) 
            elif len(img.shape) == 2:
                #image must have 3 dimmensions
                img = img.unsqueeze(0)
            elif len(img.shape) == 3:

                #Use when rgb should be bgr
                # for i in range(len(img)):
                #     img[i] = img[i][..., [2, 1, 0]]
                img = img.permute(2,0,1)

            #model wants arrays as it would be in batches. Creating another dimension as the batch dimension
            img = img.unsqueeze(0)
            # Inference
            t1 = time_synchronized()

            with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                pred = model(img, augment=False)[0]
            t2 = time_synchronized()

            # Apply NMS
            pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
            t3 = time_synchronized()

            numpy_img = img.cpu().detach().numpy()[0][0]*255
            # Process detections
            for det in pred:  # detections per image
                s=''
                gn = torch.tensor(img.shape[2:])[[1, 0, 1, 0]]  # normalization gain whwh
                if len(det):
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        numpy_xyxy = [tensor.detach().cpu().numpy() for tensor in xyxy]
                        confidence = conf.detach().cpu().numpy().item()
                        point_detection, bbox_detection = gp.convert_to_geopackage_item(numpy_xyxy, meter_per_pixel, image_gps_coordinates, confidence, names[int(cls)])
                        points.append(point_detection)
                        bounding_boxes.append(bbox_detection)
                        if save_txt or build_dataset_path != None:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            with open(combined_image_path.split('.')[0] + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                        if save_image:  # Add bbox to image
                            label = f'{names[int(cls)]} {conf:.2f}'
                            plot_one_box(xyxy, numpy_img, label=label, color=colors[int(cls)], line_thickness=1)


                log.info(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
            if not save_image and build_dataset_path == None:
                #remove all downloaded images
                os.remove(combined_image_path)
                [os.remove(image_path) for image_path in channels]
            else:
                if build_dataset_path and os.path.isfile(combined_image_path.split('.')[0] + '.txt'):
                    os.replace(combined_image_path, os.path.join(build_dataset_path,'images', os.path.basename(combined_image_path)))
                    os.replace(combined_image_path.split('.')[0] + '.txt', os.path.join(build_dataset_path,'labels', os.path.basename(combined_image_path.split('.')[0] + '.txt')))

                if(opt.write_to_tmp):
                    ip.write_geotif_at_location(channels[0], '/tmp/{},{},{},{}.tiff'.format(y,x,y+geographical_image_size,x+geographical_image_size),  np.array([numpy_img]))
                    shutil.copy('/tmp/{},{},{},{}.tiff'.format(y,x,y+geographical_image_size,x+geographical_image_size),'{}/{},{},{},{}.tiff'.format(output,y,x,y+geographical_image_size,x+geographical_image_size))
                else:
                    ip.write_geotif_at_location(channels[0], '{}/{},{},{},{}.tiff'.format(output,y,x,y+geographical_image_size,x+geographical_image_size),  np.array([numpy_img]))

                if len(points) > 0:
                    gdf_points = gp.create_point_dataframe(points) 
                    gdf_bboxs = gp.create_bbox_dataframe(bounding_boxes)           
                    if(opt.write_to_tmp):
                        gp.create_gpkg(gdf_points, gdf_bboxs, "/tmp/{}.gpkg".format(coordinates))
                        shutil.copy('/tmp/{}.gpkg'.format(coordinates), "{}/{}.gpkg".format(output,coordinates))
                    else:
                        gp.create_gpkg(gdf_points, gdf_bboxs, "{}/{}.gpkg".format(output,coordinates))
    log.info(f'Done. ({time.time() - t0:.3f}s)')
    
    #extra goodies, split the built dataset to train and validation
    if build_dataset_path:
        yc.split_yolo_data(build_dataset_path, 0.2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--coordinates', type=str, help='coordinates in which the detector should be applied to. ex: 587500,6563500,588500,6565500')
    parser.add_argument('--geographical_image_size', action='store', type=int, default=1000, help='the size for each tif image, ex 1000 (Meter)')
    parser.add_argument('--offset', action='store', type=int, default=0, help='the offset for each sample. Usen when gather geographically spread training data')
    parser.add_argument('--configuration', action='store', type=str, default='hillshade')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--output', action='store', type=str, default='output')
    parser.add_argument('--write_to_tmp', action='store_true', help='if output is network file share, set this to True')
    parser.add_argument('--save_image', action='store_true')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--init_whitebox', action='store_true')
    parser.add_argument('--build_dataset_path', action='store', default=None, help='auto annotate a new tranining dataset. define the path to store the data')
    parser.add_argument('--algorithm', type=str, default='yolov7', help='current available algorithms: yolov7, yolor')
    parser.add_argument('--yolor_cfg', type=str, help='path to yolor.cfg')
    parser.add_argument('--names', default=['obj'], nargs='+', type=str, help="class names")
    opt = parser.parse_args()
    log.info(opt)

    with torch.no_grad():
        detect()
