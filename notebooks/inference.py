import os
import numpy as np
import tifffile
from osgeo import gdal
import numpy as np
import sys
sys.path

from osgeo import gdal
from osgeo import osr


def write_gtiff(array, gdal_obj, outputpath, dtype=gdal.GDT_UInt16, options=0, color_table=0, nbands=1, nodata=False):
    """
    Writes a geotiff.

    array: numpy array to write as geotiff
    gdal_obj: object created by gdal.Open() using a tiff that has the SAME CRS, geotransform, and size as the array you're writing
    outputpath: path including filename.tiff
    dtype (OPTIONAL): datatype to save as. use gdal.GDT_Float32 for floating point and gdal.GDT_Byte for 4 bit int: https://gdal.org/drivers/raster/gtiff.html
    select from Byte, UInt16, Int16, UInt32, Int32, Float32, Float64, CInt16, CInt32, CFloat32 and CFloat64
    nodata (default: FALSE): set to any value you want to use for nodata; if FALSE, nodata is not set
    """

    gt = gdal_obj.GetGeoTransform()

    width = np.shape(array)[1]
    height = np.shape(array)[0]

    # Prepare destination file
    driver = gdal.GetDriverByName("GTiff")
    if options != 0:
        dest = driver.Create(outputpath, width, height, nbands, dtype, options)
    else:
        dest = driver.Create(outputpath, width, height, nbands, dtype)

    # Write output raster
    if color_table != 0:
        dest.GetRasterBand(1).SetColorTable(color_table)

    dest.GetRasterBand(1).WriteArray(array)

    if nodata is not False:
        dest.GetRasterBand(1).SetNoDataValue(nodata)

    # Set transform and projection
    dest.SetGeoTransform(gt)
    wkt = gdal_obj.GetProjection()
    srs = osr.SpatialReference()
    srs.ImportFromWkt(wkt)
    dest.SetProjection(srs.ExportToWkt())

    # Close output raster dataset
    dest = None


def patchify_x(img, start_y, patches, tile_size, margin, width):
    start_x = 0
    _, _, bands = img.shape
    while start_x + tile_size <= width:
        patches.append(img[start_y:start_y+tile_size,
                           start_x:start_x+tile_size, :].copy())
        # subtract own margin and margin of previous patch, so that margin of
        # previous patch is covered by the active area of the next patch
        start_x += tile_size - 2 * margin
        assert patches[-1].shape == (tile_size, tile_size, bands),\
            'shape: {}'.format(patches[-1].shape)
    # handle right boarder
    if start_x < width:
        start_x = width - tile_size
        patches.append(img[start_y:start_y+tile_size,
                           start_x:start_x+tile_size, :].copy())
        assert patches[-1].shape == (tile_size, tile_size, bands),\
            'shape: {}'.format(patches[-1].shape)


def patchify(img, tile_size, margin):
    patches = []

    height, width, bands = img.shape
    start_y = 0
    while start_y + tile_size <= height:
        patchify_x(img, start_y, patches, tile_size, margin, width)
        start_y += tile_size - 2 * margin
    # handle bottom boarder
    if start_y < height:
        start_y = height - tile_size
        patchify_x(img, start_y, patches, tile_size, margin, width)

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
    img = np.zeros(shape[:-1])
    height, width, _ = shape
    remain_height = height % tile_size
    remain_width = width % tile_size

    dest_start_y = 0
    dest_start_x = 0

    for i, patch in enumerate(patches):
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


def read_input(bands):
    '''Assemble input from list of provided tif files
       inputs will be added in order in which they are provided
    Parameters
    ----------
    bands : list of pathes to tif files
    Returns
    -------
    Tensor of shape (input height, input width, number of bands)
    '''
    tmp = tifffile.imread(bands[0])
    img = np.zeros([*tmp.shape, len(bands)])
    for i, band in enumerate(bands):
        tmp = tifffile.imread(band)
        tmp = tmp.astype(np.float32)
        img[:, :, i] = tmp

    return img


#def main(input_path, model_path, out_path, img_type, tile_size, margin,
#         threshold, wo_crf):
def main(img_path, model_path, out_path, img_type, tile_size, margin, 
         depth, class_num):
    #load model

    model = keras.models.load_model(model_path)# load old model
    
    # setup paths
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

    for bands in zip(*imgs):
        predicted = []

        img = read_input(bands)

        # we do not need to patchify image if image is too small to be split
        # into patches - assume that img width == img height
        do_patchify = True if tile_size < img.shape[0] else False

        if do_patchify:
            patches = patchify(img, tile_size, margin)
        else:
            patches = [img]

        # find suitable batch size
        for i in [8, 4, 2, 1]:
            if len(patches) % i == 0:
                bs = i
                break

        # perform prediction
        for i in range(0, len(patches), bs):
            batch = np.array(patches[i:i+bs])
            batch = batch.reshape((bs, *input_shape))
            out = model.predict(batch)
            for o in out:
                # choose id of output band with maximum probability
                tmp = np.argmax(o, axis=-1)
                predicted.append(tmp.reshape(input_shape[:-1]))

        if do_patchify:
            out = unpatchify(img.shape, predicted, tile_size, margin)
        else:
            out = predicted[0]

        # write image
        img_name = os.path.basename(bands[0]).split('.')[0]
        InutFileWithKnownExtent = gdal.Open(bands[0])
        utils.WriteGeotiff.write_gtiff(out, InutFileWithKnownExtent,
                                       os.path.join(out_path,
                                                    '{}.{}'.format(img_name,
                                                                   img_type)))


