import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import os
import logging

logging.basicConfig(level='INFO')
log = logging.getLogger()


def convert_to_geopackage_item(numpy_xyxy: list, meter_per_pixel: list, image_gps_coordinates:list, confidence: float, class_name: str) -> tuple:
    '''numpy_xyxy: Takes a list of image coordinates that comes from object detection. 
    meter_per_pixel is a list x and y pixel representation meter per pixel.
    image_gps_coordinates is the image upper left coordinates.
    confidence is the AI models prediction confidence.
    class_name is the prediction class.
    Returns geo point and geo bounding box'''
 
    min_x, min_y, max_x, max_y = numpy_xyxy
    longitude = image_gps_coordinates[3][0]
    latitude = image_gps_coordinates[4][0]
    min_longitude = (int)(longitude + min_x * meter_per_pixel[0][0])
    max_longitude = (int)(longitude + max_x * meter_per_pixel[0][0])
    min_latitude = (int)(latitude - min_y * meter_per_pixel[1][0])
    max_latitude = (int)(latitude - max_y * meter_per_pixel[1][0])

    x_point = (min_longitude+max_longitude)/2
    y_point = (min_latitude+max_latitude)/2

    point_detection = [class_name, x_point, y_point, confidence, 'model']
    bbox_detection = [class_name, min_longitude, min_latitude, max_longitude, max_latitude, confidence, 'model']
    #pdb.set_trace()
    return point_detection, bbox_detection


def create_point_dataframe(annotations: list) -> gpd.GeoDataFrame:
    '''Takes a list of geo points. ex: [['obj', 510039.5, 6441316.5, 0.48598846793174744, 'model']].
    returns a geodataframe with the geo points'''
    x_points = []
    y_points = []
    classes_string = []
    probs = []
    models = []
    for annotation in annotations:
        class_string, x_point, y_point, prob, model_name = annotation
        classes_string.append(class_string)
        x_points.append(x_point)
        y_points.append(y_point)
        probs.append(prob)
        models.append(model_name)
    df = pd.DataFrame({    
        'class': classes_string,
        'detection_probability': probs,
        'model_used': models,
        'POINT_X': x_points,
        'POINT_Y': y_points,})

    geopandas_data_frame = gpd.GeoDataFrame(df, crs='EPSG:3006', geometry=gpd.points_from_xy(df.POINT_X, df.POINT_Y))
    return geopandas_data_frame


def create_bbox_dataframe(annotations: list) -> gpd.GeoDataFrame:
    '''Takes a list of geo bounding boxes. ex: [['obj', 510020, 6441336, 510059, 6441297, 0.48598846793174744, 'model']].
    returns a geodataframe with the geo bounding boxes'''
    class_strings = []
    bounding_boxes = []
    probs = []
    models = []

    for annotation in annotations:
        x_positions = []
        y_positions = []
    
        class_string, xmin, ymin, xmax, ymax, prob, model_name = annotation
        class_strings.append(class_string)
        arcgis_ordered_positions= [[xmin,ymin], [xmin,ymax], [xmax,ymax], [xmax,ymin], [xmin,ymin]]
        for position in arcgis_ordered_positions:
            x_position = position[0]
            y_position = position[1]
            x_positions.append(x_position)
            y_positions.append(y_position)
        probs.append(prob)
        models.append(model_name)
        bounding_box_polygon = Polygon(zip(x_positions, y_positions))
        bounding_boxes.append(bounding_box_polygon)
    df = pd.DataFrame({'class': class_strings,
                        'detection_probability': probs,
                        'model_used': models,
                        })
    geopandas_data_frame = gpd.GeoDataFrame(df, crs='EPSG:3006', geometry=bounding_boxes)
    return geopandas_data_frame



def create_gpkg(geopandas_points_data_frame: gpd.GeoDataFrame, geopandas_bbox_data_frame: gpd.GeoDataFrame, destination_path: str):
    '''Takes 2 geodataframes and a destination path to save the data into a .gpkg file'''
    if os.path.isfile(destination_path):
        log.info("appending to file " + destination_path)
        geopandas_points_data_frame.to_file(driver='GPKG', filename=destination_path, layer='CenterPoints', encoding='utf-8', mode='a')
        geopandas_bbox_data_frame.to_file(driver='GPKG', filename=destination_path, layer='BoundingBoxes', encoding='utf-8',mode='a')
    else:
        log.info("writing")
        geopandas_points_data_frame.to_file(driver='GPKG', filename=destination_path, layer='CenterPoints', encoding='utf-8', mode='w')
        geopandas_bbox_data_frame.to_file(driver='GPKG', filename=destination_path, layer='BoundingBoxes', encoding='utf-8',mode='w')


def save_bounding_boxes(geodf_separata_polygoner: gpd.GeoDataFrame, destination_path: str, layer='BoundingBoxes'):
    '''Take a geodataframe of bounding boxes and a destination_path to save them into a .gpkg file'''
    polys = []
    for poly in geodf_separata_polygoner['geometry']:
        polys.append(poly)
        
    f = gpd.GeoDataFrame(geodf_separata_polygoner,geometry=polys,crs='EPSG:3006')

    f.to_file(driver='GPKG', filename=destination_path, layer=layer, encoding='utf-8', mode='w')


def read_gpkg(file_path: str, row=-1, layer=None):
    '''Takes a file path to a geo file, ex. .gpkg.
    Returns it as a dataframe'''
    points_df = gpd.read_file(file_path,rows=row, layer=layer)
    points_df = points_df.set_crs(epsg=3006)
    return  points_df


def merge_geopackages(input_folder: str, output_file: str):
    '''Takes an input dir path where .gpks are located.
        Takes a name for the output file'''
    merged_gdf = gpd.GeoDataFrame()
    

    for file in os.listdir(input_folder):
        if file.endswith(".gpkg"):
            file_path = os.path.join(input_folder, file)
            gdf = gpd.read_file(file_path, driver="GPKG")
            merged_gdf = merged_gdf.append(gdf, ignore_index=True)
    
    merged_gdf.to_file(os.path.join(input_folder,output_file), driver="GPKG")