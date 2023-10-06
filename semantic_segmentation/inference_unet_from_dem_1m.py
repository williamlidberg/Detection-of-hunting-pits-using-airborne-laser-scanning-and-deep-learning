import os
import shutil
import numpy as np
import tifffile
from osgeo import gdal
from osgeo import ogr
import geopandas as gpd
import utils.unet
import utils.WriteGeotiff
import os
import argparse
import tifffile
import whitebox

whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()

# hardcoded for demonstration purposes
inference_path = '/workspace/temp_inference/'

# def profile_curvature(temp_dir, input_path, normalized_profile_curvature, extent):
#     wbt.profile_curvature(
#         dem = input_path, 
#         output = temp_dir + os.path.basename(input_path), 
#         log=False, 
#         zfactor=None)
#     img = tifffile.imread(temp_dir + os.path.basename(input_path))
#     img[img > 1] = 1 
#     img[img < -2] = -2
#     normed_profile_curvature = (img--2)/(1--2)
#     utils.WriteGeotiff.write_gtiff(normed_profile_curvature, extent, normalized_profile_curvature, gdal.GDT_Float32)

def minimal_curvature(temp_dir, input_path, normalized_minimal_curvature, extent):
    wbt.minimal_curvature(
        dem = input_path, 
        output = temp_dir + os.path.basename(input_path), 
        log=True, 
        zfactor=None)
    img = tifffile.imread(temp_dir + os.path.basename(input_path))
    img[img > 5] = 5
    img[img < -5] = -5
    normed_minimal_curvature = (img--5)/(5--5)
    utils.WriteGeotiff.write_gtiff(normed_minimal_curvature, extent, normalized_minimal_curvature, gdal.GDT_Float32)


def patchify_x(img, start_y, patches, tile_size, margin, width, channel_last):
    start_x = 0
    while start_x + tile_size <= width:
        if channel_last:
            patches.append(img[start_y:start_y+tile_size,
                               start_x:start_x+tile_size, :].copy())
        else:
            patches.append(img[:, start_y:start_y+tile_size,
                               start_x:start_x+tile_size].copy())
        # subtract own margin and margin of previous patch, so that margin of
        # previous patch is covered by the active area of the next patch
        start_x += tile_size - 2 * margin
    # handle right boarder
    if start_x < width:
        start_x = width - tile_size
        if channel_last:
            patches.append(img[start_y:start_y+tile_size,
                               start_x:start_x+tile_size, :].copy())
        else:
            patches.append(img[:, start_y:start_y+tile_size,
                               start_x:start_x+tile_size].copy())


def patchify(img, tile_size, margin, channel_last):
    patches = []

    if channel_last:
        height, width, _ = img.shape
    else:
        _, height, width = img.shape
    start_y = 0
    while start_y + tile_size <= height:
        patchify_x(img, start_y, patches, tile_size, margin, width,
                   channel_last)
        start_y += tile_size - 2 * margin
    # handle bottom boarder
    if start_y < height:
        start_y = height - tile_size
        patchify_x(img, start_y, patches, tile_size, margin, width,
                   channel_last)

    return patches


def start_and_end(base, tile_size, margin, limit, remainder):
    if base == 0:
        src_start = 0
        src_end = tile_size - margin
    elif base + (tile_size - margin) > limit:
        src_start = tile_size - remainder
        src_end = tile_size
    else:
        src_start = margin
        src_end = tile_size - margin

    return src_start, src_end


def unpatchify(shape, patches, tile_size, margin):
    img = np.zeros(shape)
    height, width = shape
    remain_height = height % tile_size
    remain_width = width % tile_size

    dest_start_y = 0
    dest_start_x = 0

    for patch in patches:
        remain_width = width - dest_start_x
        remain_height = height - dest_start_y
        src_start_y, src_end_y = start_and_end(dest_start_y, tile_size, margin,
                                               height, remain_height)
        src_start_x, src_end_x = start_and_end(dest_start_x, tile_size, margin,
                                               width, remain_width)
        y_length = src_end_y - src_start_y
        x_length = src_end_x - src_start_x
        img[dest_start_y:dest_start_y+y_length,
            dest_start_x:dest_start_x+x_length] = patch[src_start_y:src_end_y,
                                                        src_start_x:src_end_x]
        dest_start_x += x_length
        if dest_start_x >= width:
            dest_start_x = 0
            dest_start_y += y_length

    return img


def read_input(bands, channel_last):
    '''Assemble input from list of provided tif files
       inputs will be added in order in which they are provided

    Parameters
    ----------
    bands : list of pathes to tif files
    channel_last : indicate location of channel dimension

    Returns
    -------
    Tensor of shape (input height, input width, number of bands) or (number of
    bands, input height, input width) - depending on channel_last

    '''
    tmp = tifffile.imread(bands[0])
    if channel_last:
        img = np.zeros([*tmp.shape, len(bands)])
    else:
        img = np.zeros([len(bands), *tmp.shape])
    for i, band in enumerate(bands):
        tmp = tifffile.imread(band)
        tmp = tmp.astype(np.float32)
        if channel_last:
            img[:, :, i] = tmp
        else:
            img[i, :, :] = tmp

    return img

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

def main(img_path, model_path, out_path, model_type, temp_dir, band_wise, depth,
         img_type, tile_size, margin, classes, output_type):
    
    # calculate profile curvature from dem
    for tile in os.listdir(img_path):
        if tile.endswith('.tif'):
            dem = img_path + tile
            print(dem)
            #normalized_profile_curvature = temp_dir + tile
            normalized_minimal_curvature = temp_dir + tile
        # calculate profile curvature from dem
            extent = gdal.Open(dem)
            #profile_curvature(temp_dir, dem, normalized_profile_curvature, extent)
            minimal_curvature(temp_dir, dem, normalized_minimal_curvature, extent)
    # setup paths
    img_path = []
    img_path.append(temp_dir)
    for path in img_path:
        if not os.path.exists(path):
            raise ValueError('Input path does not exist: {}'.format(path))
    # assume that either folder or image is given for all channels
    if os.path.isdir(img_path[0]):
        imgs = []
        for path in img_path:
            tmp = [os.path.join(path, f) for f in os.listdir(path)
                   if not f.startswith('._') and f.endswith('.tif')]
            imgs.append(tmp)
    else:
        imgs = [[f] for f in img_path]
    #imgs = [[f] for f in temp_dir]
   # load model
    model_cls = utils.unet.MODELS[model_type]

    # if model_cls.CHANNEL_LAST:
    #     input_shape = (tile_size, tile_size, 1)
    # else:
    #     input_shape = (1, tile_size, tile_size)
    input_shape = (tile_size, tile_size, 1)
    model = model_cls(
               input_shape, depth=depth,
               classes=len(classes.split(',')),
               entry_block=not band_wise,
               weighting=utils.unet.SegmentationModelInterface.WEIGHTING.NONE)
    model.load_weights(model_path)

    for bands in zip(*imgs):
        predicted = []

        img = read_input(bands, model.CHANNEL_LAST)

        # we do not need to patchify image if image is too small to be split
        # into patches - assume that img width == img height
        do_patchify = tile_size < img.shape[0]

        if do_patchify:
            patches = patchify(img, tile_size, margin, model.CHANNEL_LAST)
        else:
            patches = [img]

        # find suitable batch size
        for i in [8, 4, 2, 1]:
            if len(patches) % i == 0:
                batch_size = i
                break

        # perform prediction
        for i in range(0, len(patches), batch_size):
            batch = np.array(patches[i:i+batch_size])
            batch = batch.reshape((batch_size, *input_shape))
            out = model.proba(batch)
            for output in out:
                # choose id of output band with maximum probability
                tmp = np.argmax(output, axis=-1)
                predicted.append(tmp.reshape(input_shape[:-1]))

        if do_patchify:
            if model.CHANNEL_LAST:
                out = unpatchify(img.shape[:-1], predicted, tile_size, margin)
            else:
                out = unpatchify(img.shape[1:], predicted, tile_size, margin)
        else:
            out = predicted[0]

        # write image
        inference_path
        img_name = os.path.basename(bands[0]).split('.')[0]
        InutFileWithKnownExtent = gdal.Open(bands[0])
        utils.WriteGeotiff.write_gtiff(out, InutFileWithKnownExtent,
                                       os.path.join(inference_path,
                                                    '{}.{}'.format(img_name,
                                                                   img_type)))
  
    
    # post processing
    min_area = 30 # from field observations
    min_ratio = -1 # from field observations
    for pred in os.listdir(inference_path):
        if pred.endswith('.tif'):
            print(pred)
            predicted = inference_path + pred
            vector_polygons = inference_path + pred.replace('.tif', 'raw.shp')
            vector_polygons_processed = inference_path + pred.replace('.tif', 'filtered.shp')
            vector_to_raster = inference_path + pred
            post_processed_prediction = out_path + pred
            post_processed_prediction_polygon = out_path + pred.replace('.tif', '.shp')

            chip = tifffile.imread(predicted)
            num_ones = (chip == 1).sum()
            if num_ones > 1:
                raster_to_polygon(predicted, vector_polygons)
                calculate_attributes(vector_polygons)
                if output_type == 'polygon':
                    delete_features(vector_polygons, min_area, min_ratio, post_processed_prediction_polygon)
                elif output_type == 'raster':
                    delete_features(vector_polygons, min_area, min_ratio, vector_polygons_processed)
                    polygon_to_raster(pred, vector_polygons_processed, vector_to_raster)
                    raster_to_binary(vector_to_raster, post_processed_prediction)                
                
            else:
                print(pred,' had no predicted labels')

    # delete temporary files
    for tile in os.listdir(temp_dir):
        if tile.endswith('tif'):
            os.remove(temp_dir + tile)
    for tile in os.listdir(inference_path):
        if tile.endswith('tif') or 'raw' in tile:
            os.remove(inference_path + tile)               

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Run inference on given '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('img_path', help='to input dem')
    parser.add_argument('model_path')
    parser.add_argument('out_path', help='Path to output folder')
    parser.add_argument('--model_type', help='Segmentation model to use',
                        choices=list(utils.unet.MODELS.keys()), default='UNet')
    parser.add_argument('--temp_dir', help='Path to temp_dir in container', default='/workspace/temp/')
    parser.add_argument('--band_wise', action='store_true',
                        help='Apply separable convolutions on input bands.')
    parser.add_argument('--depth', type=int, default=2)
    parser.add_argument('--img_type', help='Output image file ending',
                        default='tif')
    parser.add_argument('--classes', help='List of class labels in ground '
                        'truth - order needs to correspond to weighting order',
                        default='0,1')
    parser.add_argument('--tile_size', help='Tile size', type=int,
                        default=250)
    parser.add_argument('--margin', help='Margin', type=int, default=100)
    parser.add_argument('--output_type', default='polygon',help='output polygon or raster')

    args = vars(parser.parse_args())
    main(**args)