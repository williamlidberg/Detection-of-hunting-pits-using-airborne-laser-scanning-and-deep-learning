import os
import whitebox
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()

labels = '/workspace/data/final_data_05m/testing/labels/'
polygon_labels = '/workspace/data/final_data_05m/testing/polygon_labels/'
for label in os.listdir(labels):
    if label.endswith('.tif'):
        image_name = labels + label
        outname = polygon_labels + label.replace('.tif', 'filtered.shp')
    wbt.raster_to_vector_polygons(
    i = image_name, 
    output = outname)


labels = '/workspace/data/final_data_1m/testing/labels/'
polygon_labels = '/workspace/data/final_data_1m/testing/polygon_labels/'
for label in os.listdir(labels):
    if label.endswith('.tif'):
        image_name = labels + label
        outname = polygon_labels + label.replace('.tif', 'filtered.shp')
    wbt.raster_to_vector_polygons(
    i = image_name, 
    output = outname)