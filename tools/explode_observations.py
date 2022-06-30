import argparse
import geopandas as gpd

def explode_shapefile(input_observations, field, output_shapefiles_dir):
    df = gpd.read_file(input_observations)
    grouped = df.groupby(field)
    for key, group in grouped:
        object_category = output_shapefiles_dir + str(key) + '.shp'
        group.to_file(object_category.format(key))


def main(input_observations, field, output_shapefiles_dir):
    explode_shapefile(input_observations, field, output_shapefiles_dir)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
                       description='Explodes/splits a shapefile based on attributes',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_observations', help = 'shapefile with observations')
    parser.add_argument('field', help = 'select attribute to use for exploding observation shapefile')
    parser.add_argument('output_shapefiles_dir', help = 'path to output shapefiles')
   
    args = vars(parser.parse_args())
    main(**args)