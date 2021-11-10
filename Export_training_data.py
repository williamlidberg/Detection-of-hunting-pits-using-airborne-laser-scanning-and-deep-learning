# This is the south
import arcpy, os
arcpy.CheckOutExtension("ImageAnalyst")
from arcpy.ia import *
from arcpy.sa import *
from arcpy import env

arcpy.env.compression = "NONE"
arcpy.env.pyramid = "NONE"
arcpy.env.overwriteOutput = True
arcpy.env.parallelProcessingFactor = "0%"


# #convert shape Labels to raster labels
# arcpy.env.snapRaster = 'D:/kolbottnar/Extract_topogrpahical_indices/Hillshade/hillshde_mosaik.tif'
# in_features = 'D:/kolbottnar/Kolbottnar_buf.shp'
# field = 'class'
# out_raster = 'D:/kolbottnar/feature_to_raster.tif'
# arcpy.conversion.FeatureToRaster(in_features, field, out_raster)
# whereClause = "VALUE is null"
# 'D:/kolbottnar/feature_to_raster_nodata.tif' = Con(out_raster, 0, 1, {where_clause})
nodatatozero = 'D:/kolbottnar/labels/nodata_to_zero.tif'
# transform the data to 8 bit thematic raster and create a attribute table which is required for the export training data for deep learning tool.
out8bit = 'D:/kolbottnar/labels/nodata_8bit.tif'
print('Copy', out8bit, 'to 8 bit unsigned')
#arcpy.CopyRaster_management(nodatatozero, out8bit,"","","","NONE","NONE","8_BIT_UNSIGNED","NONE","NONE", "TIFF", "NONE", "", "")
print('Set raster properties to THEMATIC')
#arcpy.SetRasterProperties_management(out8bit, "THEMATIC","", "#", "#")
print('Build Raster attribute table')
#arcpy.BuildRasterAttributeTable_management(out8bit, "Overwrite")
print('Add field')
#arcpy.AddField_management(out8bit, 'ClassValue', "TEXT")
print('Calculate field')
#arcpy.CalculateField_management(out8bit, 'ClassValue', "!Value!","PYTHON3")
print('Done')



# These are the high HighPassMedianFilter
inRaster = 'D:/kolbottnar/Extract_topogrpahical_indices/Hillshade/hillshde_mosaik.tif'
#This are the labels
in_training = out8bit
# areas where training data will be exported from
trainingtiles = "Y:/William/DeepLearning/DitchnetProduction/Digitaliserade_rutor_20191202/Digitaliserade_rutor_20191202.shp"
# These are the image chips used for training
out_folder_training = 'Y:/William/Kolbottnar/data/hillshade'

# Export training data
image_chip_format = "TIFF"
tile_size_x = "512"
tile_size_y = "512"
stride_x= "512" # Stride means that the tiles will have overlap in both x and y directions
stride_y= "512"
output_nofeature_tiles= "ONLY_TILES_WITH_FEATURES"
metadata_format= "Classified_Tiles"
start_index = 0
classvalue_field = 'Value'
buffer_radius = 0 # This is somehing to consider. Right now the model trains untill the very edge of the tiles
in_mask_polygons = '' # only data within these polygons will be exported. This is to avoid exporting large areas of no data.
rotation_angle = 0 # a rotation of 90 degrees will be applied to the training data
reference_system = "MAP_SPACE"
processing_mode = "PROCESS_AS_MOSAICKED_IMAGE"
blacken_around_feature = "NO_BLACKEN"
crop_mode = ""

print('exporting training data')
# Execute
ExportTrainingDataForDeepLearning(inRaster, out_folder_training, in_training,
    image_chip_format,tile_size_x, tile_size_y, stride_x,
    stride_y,output_nofeature_tiles, metadata_format, start_index,
    classvalue_field, buffer_radius, in_mask_polygons, rotation_angle,
    reference_system, processing_mode, blacken_around_feature, crop_mode)
