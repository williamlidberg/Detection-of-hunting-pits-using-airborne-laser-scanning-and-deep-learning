import os
import argparse
import shutil
import tifffile
import os
os.environ["WBT_LINUX"] = "MUSL"
#import whitebox
import whitebox
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()
from osgeo import ogr
import geopandas as gpd
parser = argparse.ArgumentParser(description='Post-process predictions to vector polygons')

def raster_to_polygon(img_name, vector_polygons):
    wbt.raster_to_vector_polygons(
    i = img_name, 
    output = vector_polygons)
    
def polygon_to_raster(basefile, filtered_poygon, post_processed_prediction):
    wbt.vector_polygons_to_raster(
    i= filtered_poygon, 
    output=post_processed_prediction, 
    field='FID', 
    nodata=False, 
    cell_size=None, 
    base=basefile)

def calculate_attributes(vector_polygons):
    # use geopandas to create area column instead of gdal
    wbt.perimeter_area_ratio(vector_polygons)
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.Open(vector_polygons,1) # 0 means read-only. 1 means writeable.

    if dataSource is None:
        print ('Could not open %s' % (vector_polygons))
    else:
        print ('Opened %s' % (vector_polygons))
        layer = dataSource.GetLayer()

    new_field = ogr.FieldDefn("Area", ogr.OFTReal)
    new_field.SetWidth(32)
    new_field.SetPrecision(2)
    layer.CreateField(new_field)

    for feature in layer:
        geom = feature.GetGeometryRef()
        area = geom.GetArea() 
        feature.SetField("Area", area)
        layer.SetFeature(feature)

def delete_features(vector_polygons, min_area, min_ratio, vector_polygons_processed):
    polygons = gpd.read_file(vector_polygons)
    mask = (polygons.Area > min_area) & (polygons.P_A_RATIO > min_ratio)
    try:
        selected_polygons = polygons.loc[mask]
        selected_polygons.to_file(vector_polygons_processed)
    except:
        print(vector_polygons,'failed for some reason')

def raster_to_binary(converted_preds, final_preds):
    wbt.greater_than(
        input1 = converted_preds, 
        input2 = 1, 
        output= final_preds, 
        incl_equals=True
    )

def main(temp_dir, input_path, output_type, output_predictions, min_area, min_ratio):
    # setup paths
    if not os.path.exists(input_path):
        raise ValueError('Input path does not exist: {}'.format(input_path))
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.endswith('.tif')]
    else:
        imgs = [input_path]

    
    for img_path in imgs:
        img_name = os.path.basename(img_path).split('.')[0]        
        vector_polygons =  os.path.join(temp_dir,'{}.{}'.format((img_name + 'raw'), 'shp'))
        vector_polygons_processed = os.path.join(temp_dir,'{}.{}'.format((img_name + 'filtered'), 'shp'))
        vector_to_raster = os.path.join(temp_dir,'{}.{}'.format(img_name, 'tif'))
        post_processed_prediction = os.path.join(output_predictions,'{}.{}'.format(img_name, 'tif'))
        post_processed_prediction_polygon = os.path.join(output_predictions,'{}.{}'.format((img_name + 'filtered'), 'shp'))

        chip = tifffile.imread(img_path)
        num_ones = (chip == 1).sum()
        if num_ones == 0:
            print(img_path,' had no predicted labels')
            raster_to_binary(img_path, post_processed_prediction)
            
        else:
            raster_to_polygon(img_path, vector_polygons)
            calculate_attributes(vector_polygons)
            if output_type == 'polygon':
                delete_features(vector_polygons, min_area, min_ratio, post_processed_prediction_polygon)
            elif output_type == 'raster':
                delete_features(vector_polygons, min_area, min_ratio, vector_polygons_processed)
                polygon_to_raster(img_path, vector_polygons_processed, vector_to_raster)
                raster_to_binary(vector_to_raster, post_processed_prediction)
                
    original_list = os.listdir(input_path)
    post_processed_list = os.listdir(output_predictions)
    for chip in original_list:
      if chip not in post_processed_list:
        org = input_path + chip
        post = output_predictions + chip
        shutil.copy(org, post)
      
    

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Extract topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('temp_dir', help= 'path to a temperary directory')
    parser.add_argument('input_path', help='Path to dem or folder of dems')
    parser.add_argument('output_predictions', help='output_predictions')
    parser.add_argument('--output_type', default='raster',help='output polygon or raster')
    parser.add_argument('--min_area', help= 'smallest detected polygon in square meters', type=int, default=20)
    parser.add_argument('--min_ratio', help= 'smallest perimiter area ratio', type=float, default=-0.3)
    args = vars(parser.parse_args())
    main(**args)