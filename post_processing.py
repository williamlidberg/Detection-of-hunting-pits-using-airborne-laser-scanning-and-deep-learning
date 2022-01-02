import os
import argparse
import whitebox
wbt = whitebox.WhiteboxTools()
from osgeo import ogr

parser = argparse.ArgumentParser(description='Extract topogrpahical incides from DEMs. ')
def calculate_area(shapefile):
    daShapefile = shapefile #path where your shape file is present 

    driver = ogr.GetDriverByName('ESRI Shapefile')

    dataSource = driver.Open(daShapefile,1) # 0 means read-only. 1 means writeable.

    # Check to see if shapefile is found.
    if dataSource is None:
        print ('Could not open %s' % (daShapefile))
    else:
        print ('Opened %s' % (daShapefile))
        layer = dataSource.GetLayer()
        featureCount = layer.GetFeatureCount()
        print ("Number of features in %s: %d" % (os.path.basename(daShapefile),featureCount))
        print ("\n")


    new_field = ogr.FieldDefn("Area", ogr.OFTReal)
    new_field.SetWidth(32)
    new_field.SetPrecision(2) #added line to set precision
    layer.CreateField(new_field)

    for feature in layer:
        geom = feature.GetGeometryRef()
        area = geom.GetArea() 
        print (area)
        feature.SetField("Area", area)
        layer.SetFeature(feature)

def main(input_path, output_polyons):

    # setup paths
    if not os.path.exists(input_path):
        raise ValueError('Input path does not exist: {}'.format(input_path))
    if os.path.isdir(input_path):
        imgs = [os.path.join(input_path, f) for f in os.listdir(input_path)
                if f.endswith('.tif')]

    else:
        imgs = [input_path]

    
    for img_path in imgs:
        predicted = []
        print(img_path)
        img_name = os.path.basename(img_path).split('.')[0]
        
        vector_polygons =  os.path.join(output_polyons,'{}.{}'.format(img_name, 'shp'))

        wbt.raster_to_vector_polygons(
            i = img_path, 
            output = vector_polygons
        )
        calculate_area(vector_polygons)
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Extract topographical indicies '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='Path to dem or folder of dems')
    parser.add_argument('output_polyons', help = 'directory to store output polyons')
    args = vars(parser.parse_args())
    main(**args)