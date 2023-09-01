import sys
sys.path.append("./yolor")
sys.path.append("./pipeline_utils")

import argparse
import torch.backends.cudnn as cudnn
from utils.plots import plot_one_box
from utils.torch_utils import select_device, time_synchronized
#from models.models import *
from YolorModels import Darknet
from utils.datasets import *
from utils.general import *
import GeoPackage as geopackage
import read_tiff_data as read_tiff_data
import numpy as np
#from PIL import Image
from tifffile import imread
import math
import matplotlib.pyplot as plt
import pathlib
import sys
import pdb

from torch.utils.data import DataLoader, TensorDataset


pltf = platform.system()
if pltf == 'Linux':
    print("on Linux system")
    pathlib.WindowsPath = pathlib.PosixPath
else:
    print("on Windows system")
    pathlib.PosixPath = pathlib.WindowsPath
#pathlib.WindowsPath = pathlib.PosixPath


__here__ = os.path.dirname(__file__)
# Add the directory of train.py to the python path to be able to laod modules. Needed after the latest AML runtime update.
sys.path.insert(1, './' + __here__)


def padding(array, xx, yy):
    """
    :param array: numpy array
    :param xx: desired height
    :param yy: desirex width
    :return: padded array
    """
    h = array.shape[0]
    w = array.shape[1]
    
    a = (xx - h) // 2
    aa = xx - a - h

    b = (yy - w) // 2
    bb = yy - b - w

    return np.pad(array, pad_width=((0, aa*2), (0, bb*2)), mode='constant')      

    
def detect(model, device, half, opt, local_pt_data_path="", local_results_path="", batch_size=8, index_x = 0, index_y = 0, combined_file=''):
    out, source, weights, view_img, save_txt, imgsz, cfg, names, filename, outfile = \
        opt.output, opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.cfg, opt.names, opt.filename, opt.outfile

    
    if combined_file == '':
        combined_file = filename
        
    tiff_file = open(os.path.join(local_pt_data_path,filename), 'rb')
    tiff_tags = read_tiff_data.get_tiff_data(tiff_file)    

    path = os.path.join(local_pt_data_path, combined_file)

    if path.endswith('.tiff') or path.endswith('.tif'): #using tifffile instead of cv2 for tiffs
        im = imread(path)
        # Ensure image has 3 dimensions
        if (len(im.shape) < 3):
            im = np.expand_dims(im, -1)

        # BGR channels TO RGB channels - assuming first three channels are the color channels
        if (im.shape[2] >= 3):
            channels_bgr = cv2.split(im)[:3]
            channels_rgb = [channels_bgr[2], channels_bgr[1], channels_bgr[0]]
            # Merge channels back into an RGB image
            img_rgb = cv2.merge(channels_rgb)
            
            if im.shape[2] > 3:
                t1 = cv2.split(im_rgb)
                t2 = cv2.split(im)[3:]
                im = cv2.merge(t1 + t2)
            else:
                im = img_rgb
        #img = img[..., ::-1]#rgb to bgr
    else:
        im = cv2.imread(path)  # BGR

    meter_per_pixel = read_tiff_data.get_meter_per_pixel(tiff_tags)
    gps_coordinate = read_tiff_data.get_gps_coordinates(tiff_tags)

    #print("meter per pixel: {}".format(meter_per_pixel))
    x_points = []
    y_points = []

    point_annotations = []
    bbox_annotations = []

    
    img_dimmension_length = len(im.shape)
    img_shape = im.shape
    tile_size = (imgsz, imgsz)
    offset = (imgsz, imgsz)
    start = time_synchronized()

    #pdb.set_trace()
 
    dataset_list = []
    cropped_images = []
    
    print("Splitting image in smaller pieces.")
    for i in range(int(math.ceil(img_shape[0] / (offset[1] * 1.0)))):
        for j in range(int(math.ceil(img_shape[1] / (offset[0] * 1.0)))):
            
            cropped_img = im[offset[1] * i:min(offset[1] * i + tile_size[1], img_shape[0]),
                          offset[0] * j:min(offset[0] * j + tile_size[0], img_shape[1])]
            
            #print(f"Cropped_image: {cropped_img.shape}")

            if img_dimmension_length == 2:
                cropped_img = padding(cropped_img, imgsz, imgsz)
 
            img0 = cropped_img
            if img_dimmension_length == 3:
                img0 = img0.transpose(2, 0, 1)

            img0 = np.ascontiguousarray(img0)

            if img_dimmension_length == 2:
                img0 = np.array([img0, img0, img0])

            desired_size = (opt.img_size, opt.img_size)
            padding_rows = desired_size[0] - img0.shape[1]
            padding_cols = desired_size[1] - img0.shape[2]

            assert padding_rows >= 0
            assert padding_cols >= 0

            img_padded = np.pad(img0, ((0, 0), (0, padding_rows), (0, padding_cols)), constant_values=(255))

            if(padding_rows > 0):
                cv2.imwrite("padded.jpg",img_padded.transpose(1, 2, 0))

            img = torch.from_numpy(img_padded).to(device)
            img = img.half() if half else img.float()  # uint8 to fp16/32
            img /= 255.0  # 0 - 255 to 0.0 - 1.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
                            
            cropped_images.append(cropped_img)
            
            dataset_list.append(img)
    
    length = int(math.ceil(img_shape[0] / (offset[1] * 1.0))) * int(math.ceil(img_shape[1] / (offset[0] * 1.0)))

    print(f"Starting inference on {length} images.")
    for batch in range(0, length, batch_size):

        data_batch = dataset_list[batch:batch+batch_size]
        
        torch_batch = torch.cat(data_batch)
        preds = model(torch_batch, augment=opt.augment)[0]
        preds = non_max_suppression(preds, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                       agnostic=opt.agnostic_nms)                      
                                       
        for index, pred in enumerate(preds):
            i = math.floor((batch + index)/int(math.ceil(img_shape[0] / (offset[1] * 1.0))))
            j = batch + index - i *  int(math.ceil(img_shape[0] / (offset[1] * 1.0)))
            true_index = i * j + j
            true_index = batch + index
            
                
            for ii, det in enumerate(pred):  # detections per image        
                if det is not None and len(det):
                   
                    if save_txt:
                        cv2.imwrite("inference/output/x_"+str(index_x) + "_y_" + str(index_y) + "_" +str(i)+str(j) + str(gps_coordinate.values[3][0]).replace('.','')+"_"+str(gps_coordinate.values[4][0]).replace('.','') +  ".jpg", cropped_images[true_index])

                    det = det[None, :]
               
                    #torch array must be copied
                    copy_det = torch.clone(det)
                    # rescale boxes from img_size to im0 size
                    copy_det[:, :4] = scale_coords(dataset_list[true_index].shape[2:], copy_det[:, :4], cropped_images[true_index].shape).round()

                    # print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s = '%g %s, ' % (n, 'snags')  # add to string

                    for *xyxy, conf, cls in det:
                        #xyxy = minx miny maxx maxy
                        min_x = xyxy[0].detach().cpu().numpy() * meter_per_pixel.values[0][0]
                        min_y = xyxy[1].detach().cpu().numpy() * meter_per_pixel.values[1][0] 
                        max_x = xyxy[2].detach().cpu().numpy() * meter_per_pixel.values[0][0]
                        max_y = xyxy[3].detach().cpu().numpy() * meter_per_pixel.values[1][0] 
                        conf = conf.detach().cpu().numpy().item()

                        if(max_x-min_x > 0.001 and max_y-min_y > 0.001):
                            min_x = (int)(gps_coordinate.values[3][0] + min_x + (tile_size[0] * j * meter_per_pixel.values[0][0]))
                            max_x = (int)(gps_coordinate.values[3][0] + max_x + (tile_size[0] * j * meter_per_pixel.values[0][0]))
                            min_y = (int)(gps_coordinate.values[4][0] - min_y - (tile_size[1] * i * meter_per_pixel.values[1][0]))
                            max_y = (int)(gps_coordinate.values[4][0] - max_y - (tile_size[1] * i * meter_per_pixel.values[1][0]))

                            x_point = (min_x+max_x)/2
                            y_point = (min_y+max_y)/2
                            point_annotations.append(['trappingpit', x_point, y_point, conf, 'model'])
                            bbox_annotations.append(['trappingpit', min_x, min_y, max_x, max_y, conf, 'model'])
                            
                            

                        #for *xyxy, conf, cls in reversed(det):
                        if save_txt:  # Write to file
                            x = [xyxy[0].detach().cpu().numpy(), xyxy[1].detach().cpu().numpy(), xyxy[2].detach().cpu().numpy(), xyxy[3].detach().cpu().numpy()]
                            y = np.copy(x)
                            y[0] = (x[0] + x[ 2]) / 2  # x center
                            y[1] = (x[ 1] + x[ 3]) / 2  # y center
                            y[2] = x[2] - x[ 0]  # width
                            y[3] = x[ 3] - x[ 1]  # height
                            y /= imgsz

                            line = (cls, *y, conf) if True else (cls, *y)  # label format
                            with open("inference/output/x_"+str(index_x) + "_y_" + str(index_y) + "_" +str(i)+str(j) + str(gps_coordinate.values[3][0]).replace('.','')+"_"+str(gps_coordinate.values[4][0]).replace('.','') + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')
                                    

    if len(point_annotations) > 0:
        print(f"create points for {len(point_annotations)} detections")
        gdf_points = geopackage.create_point_dataframe(point_annotations)
        print(gdf_points)
        print("create bboxes")
        gdf_bboxs = geopackage.create_bbox_dataframe(bbox_annotations)
        print("create gpkg file: {}".format(os.path.join('./inference/',outfile)))
        
        # Create the geopackage file in /tmp/ to avoid creating it on a volume that is mapped to a network share which gives a strange exception in gdal.
        geopackage.create_gpkg(gdf_points, gdf_bboxs, os.path.join("./inference/",outfile))
        point_annotations = []
        bbox_annotations = []                  

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolor_best_overall.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='inference/output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=128, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--cfg', type=str, default='cfg/yolor_p6.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='../data.names', help='*.cfg path')
    parser.add_argument('--filename', type=str, default='input.tif', help='*.tif')
    parser.add_argument('--outfile', type=str, default='out.gpkg', help='*.gpkg')
    parser.add_argument('--batch-size', type=int, default=8, help='number of images to detect in a batch',action='store', dest='batch_size')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                strip_optimizer(opt.weights)
        else:
            detect(opt)
