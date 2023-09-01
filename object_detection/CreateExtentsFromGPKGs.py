import os
import argparse
from  AIRasterDataprocessing.createcontainingextents.coarse_clustring import coarse_clustering_of_points
from  AIRasterDataprocessing.createcontainingextents.point_clustring import point_clustering_to_boxes
import geopandas as geopd
import pandas as pd
import sys
sys.path.append("pipeline_utils")
import GeoPackage as gp
import logging

import pdb

def convert_polygon(geodf_polygoner: tuple) -> tuple:
    '''Takes a Geopandas data frame. Converts all Multipolygons to Polygon and returns the data frame'''
    lista = []
    
    for polygon in geodf_polygoner.iterrows():
        polygon1 = polygon[1]
        if polygon1['geometry'].geom_type == 'MultiPolygon':
            # extract polygons out of multipolygon
            for p1 in polygon1['geometry'].geoms:
                p = {'geometry':p1}
                lista.append(p)
        elif polygon1['geometry'].geom_type == 'Polygon':
            p = {'geometry':polygon1['geometry']}
           
            lista.append(p)
    df_separata_polygoner = pd.DataFrame(lista)
    all_polygons = geopd.GeoDataFrame(df_separata_polygoner) 
    return all_polygons

def main(opt):

    logging.basicConfig(level='INFO')
    log = logging.getLogger()

    files = []
    for root, dirs, filenames in os.walk(opt.input_path):
        for filename in filenames:
            print(filename)
            log.info(f'Using file {filename}')
            files.append(os.path.join(root,filename))

    combined = []
    for f in files:
        if opt.convert_to_bbox:
            polygons = gp.read_gpkg(f)
            if 'class' not in polygons.columns:
                polygons = convert_polygon(polygons)
                polygons['class']='obj'
                polygons.to_file(driver='GPKG', filename=f, layer=None, encoding='utf-8', mode='w')
        else:
            polygons = gp.read_gpkg(f, layer='BoundingBoxes')
        log.info(f'Reading bounding box layer from  {f}')
        polygons.drop(polygons.columns.difference(['geometry']),1, inplace=False)

        combined.append(polygons)

    combined = pd.concat(combined)
    combined_filename = os.path.join(opt.output_path, 'combined.gpkg')
    gp.save_bounding_boxes(combined, combined_filename)

    max_n_in_cluster = 1000
    #Gör först en grov clustringsalgorithm på alla polygoner
    log.info(f'Maximum number of polygons per cluster {max_n_in_cluster}')
    log.info(f'Creating clusters from {combined_filename}')
    coarse_clustering_of_points(combined_filename, opt.output_path, max_n_in_cluster)

    #Gör en finare uppdelning utifrån den grova indelningen
    log.info(os.path.join(opt.output_path,'*coarse.gpkg'))
    point_clustering_to_boxes(os.path.join(opt.output_path,'*_coarse.gpkg'), opt.output_path, im_size=int(opt.extent_size), write_eval=False)
    
    log.info('Remove combined files')
    os.remove(os.path.join(opt.output_path, 'combined.gpkg'))
    os.remove(os.path.join(opt.output_path, 'combined_coarse.gpkg'))

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', action='store', default='geopackage/small')
    parser.add_argument('--output_path', action='store', default='geopackage')
    parser.add_argument('--extent_size', action='store', default='1280', help='size in meters for extent')
    parser.add_argument('--convert_to_bbox', action='store_true', help='Set flag if input polygons is not of bounding box type')
    opt = parser.parse_args()

    main(opt)