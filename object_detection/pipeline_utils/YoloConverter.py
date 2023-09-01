import Common as co
import os
from sklearn.model_selection import ShuffleSplit
import logging

logging.basicConfig(level='INFO')
log = logging.getLogger()
from shapely.geometry import Polygon
import geopandas as gpd
import pandas as pd

def get_images_from_path(path: str) -> list:
    '''Compile a list of all data images
    Search through all folders and subfolders of the master folder,
    save the path of all images encountered'''
    images = []
    for dirpath, _, filenames in os.walk(os.path.join(path, 'images')):
        for filename in filenames:
            file_path = '{}/{}'.format(dirpath, filename)
            if filename.lower().endswith(".tiff") or filename.lower().endswith(".tif"):
                images.append(file_path)
    return images


def write_txt_file(images: list, path: str, filename: str, indexes: list):
    '''Takes a list of images. A path to the folder of text filesto write the split text file.
    The filename can be train or valid. The indexes is samples to be picked from images
    '''
    with open("{}/{}.txt".format(path, filename), "w") as outfile:
        for index in indexes:
            image_path = images[index].replace('\\', '/')
            outfile.write(image_path)
            outfile.write("\n")
        outfile.close()


def split_yolo_data(path: str, validation_fraction=0.2):
    '''Takes a path to all the images. 
    Validation_fraction is the amount ratio of the validation compared to the train data'''
    images = get_images_from_path(path)
    log.info(f'Number of images found in dataset: {len(images)}')
    shuffle_split = ShuffleSplit(n_splits=1, test_size=validation_fraction)
    folds = list(shuffle_split.split(images))
    split_dir = os.path.join(path, 'split')
    co.ensure_dir(split_dir)

    for index, fold in enumerate(folds):
        fold_path = '{}'.format(split_dir, (index+1))
        train = fold[0]
        val = fold[1]
        write_txt_file(images, fold_path, 'train', train)
        write_txt_file(images, fold_path, 'val', val)


def create_data_yaml(path: str, classes: list):
    '''Creates the data.yaml file for the built training dataset.
    Takes the path to the training dataset as a string. Takes a list of classes names as a list.'''
    with open('{}/data.yaml'.format(path), 'w') as outfile:
        outfile.write('train: {}/split/train.txt \n'.format(path))
        outfile.write('val: {}/split/val.txt \n\n'.format(path))
        outfile.write('nc: {}\n'.format(len(classes)))
        outfile.write('names: {}\n'.format(classes))
    outfile.close()

    
def xyxy2xywh_list(x: list):
    '''Takes a list with numbers. Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right'''
    y = x.copy()
    y[0] = (x[0] + x[2]) / 2  # x center
    y[1] = (x[1] + x[3]) / 2  # y center
    y[2] = x[2] - x[0]  # width
    y[3] = x[3] - x[1]  # height
    return y


def convert_polygons_to_yolo(image_polygon: tuple, original_polygon: tuple) -> list:
    '''Takes Geopandas 2 Geopandas dataframe as input. Returns the yolo format values x,y,w,h as a list'''
    image_polygon_bounds = image_polygon[1].geometry.bounds
    image_width = image_polygon_bounds[2] - image_polygon_bounds[0]
    image_height = image_polygon_bounds[3] - image_polygon_bounds[1]

    #get the intersecion bounds between the polygons
    minx, miny, maxx, maxy = image_polygon[1].geometry.intersection(original_polygon[1].geometry).bounds
    #normalize the coordinates to the image polygon
    x1 = (minx - image_polygon_bounds[0]) / image_width 
    x2 = (maxx - image_polygon_bounds[0]) / image_width
    y2 = 1.0-(miny - image_polygon_bounds[1]) / image_height
    y1 = 1.0-(maxy - image_polygon_bounds[1]) / image_height
    #convert the coordinates to yolo format x center, y center, width, height
    xywh = xyxy2xywh_list([x1,y1,x2,y2])
    return xywh


def convert_yolo_to_polygon(yolo_labels_path: str, output: str):
    '''Converts YOLO format bounding box annotations to polygons and saves them as a GeoPackage.

    Args:
        yolo_labels_path (str): Path to the folder containing the YOLO format annotation files.
        output (str): Path to the output GeoPackage file.

    Returns:
        None.

    The YOLO format annotation files must have the same name as the corresponding image files, but with a '.txt' extension. The format of each line in the YOLO format annotation file is as follows:

        <class_id> <x_center> <y_center> <width> <height>

    The <class_id> is an integer representing the object class. The <x_center> and <y_center> are float values representing the center of the bounding box relative to the width and height of the image. The <width> and <height> are float values representing the width and height of the bounding box relative to the width and height of the image.

    The function reads the image dimensions from the name of the annotation file and converts the YOLO format coordinates to pixel coordinates. Then, it creates a shapely Polygon object from the pixel coordinates and stores it with the class ID in a dictionary. Finally, it creates a GeoDataFrame from the list of dictionaries and saves it as a GeoPackage with a 'BoundingBoxes' layer.
    '''
    features = []
    x_points = []
    y_points = []
    classes = []
    # Loop over each file in the folder
    for txt_file in os.listdir(yolo_labels_path):
        if not txt_file.endswith(".txt"):
            continue
        # Extract the image dimensions from the file name, if possible
        try:
            name = os.path.splitext(txt_file)[0]
            x_min, y_min, x_max, y_max = [int(coord) for coord in name.split('_')[1][1:-1].split(',')]
        except (IndexError, ValueError):
            # Skip this file if the expected coordinates are not present in the filename
            continue
        # Read the yolo annotation file
        with open(os.path.join(yolo_labels_path, txt_file), "r") as f:
            for line in f:
                annotation = line.split()
                # convert yolo coordinates to pixel coordinates using the image dimensions from the file name
                class_id, x_center, y_center, width, height = [float(x) for x in annotation[0:5]]
                x_min_rect = int((x_center - width / 2) * (x_max - x_min) + x_min)
                y_min_rect = int((1 - y_center - height / 2) * (y_max - y_min) + y_min)
                x_max_rect = int((x_center + width / 2) * (x_max - x_min) + x_min)
                y_max_rect = int((1 - y_center + height / 2) * (y_max - y_min) + y_min)
                # create a shapely Polygon object from the rectangle coordinates
                x_positions=[]
                y_positions=[]
                arcgis_ordered_positions = [[x_min_rect,y_min_rect], [x_min_rect,y_max_rect], [x_max_rect,y_max_rect], [x_max_rect,y_min_rect], [x_min_rect,y_min_rect]]
                for position in arcgis_ordered_positions:
                    x_position = position[0]
                    y_position = position[1]
                    x_positions.append(x_position)
                    y_positions.append(y_position)
                poly = Polygon(zip(x_positions, y_positions))

                # create a dictionary with the feature information and add it to the list of features
                feature = {"geometry": poly, "class": class_id}
                features.append(feature)
                x_points.append((x_min_rect + x_max_rect)/2)
                y_points.append((y_min_rect + y_max_rect)/2)
                classes.append(class_id)

    df = pd.DataFrame({'class': classes,'POINT_X': x_points,'POINT_Y': y_points,})
    gdf_points = gpd.GeoDataFrame(df, crs='EPSG:3006', geometry=gpd.points_from_xy(df.POINT_X, df.POINT_Y))

    # create a geopandas dataframe from the list of features
    gdf = gpd.GeoDataFrame(features)
    gdf.to_file(output, crs='EPSG:3006', driver='GPKG', encoding='utf-8', mode='w', layer='BoundingBoxes')
    gdf_points.to_file(driver='GPKG', filename=output, crs='EPSG:3006', layer='CenterPoints', encoding='utf-8', mode='w')