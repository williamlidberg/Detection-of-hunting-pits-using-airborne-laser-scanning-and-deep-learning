import sys
sys.path.append("pipeline_utils")
import GeoPackage as gp
import geopandas as geopandas
from shapely.geometry import box
import argparse

def write_coordinates_to_file(x1: int, y1: int, x2: int, y2: int, num_parts: int,
                              BatchInferenceCommand: str, filename: str) -> None:
    """
    Write the coordinates of the given rectangular region to a file.

    Args:
        x1 (int): The x-coordinate of the top-left corner of the region.
        y1 (int): The y-coordinate of the top-left corner of the region.
        x2 (int): The x-coordinate of the bottom-right corner of the region.
        y2 (int): The y-coordinate of the bottom-right corner of the region.
        num_parts (int): The number of parts to divide the region into.
        BatchInferenceCommand (str): The batch inference command to be executed on the coordinates.
        filename (str): The name of the file to write the coordinates to.

    Returns:
        None: This function does not return anything, it just writes to a file.

    """
    width = x2 - x1
    height = y2 - y1

    part_width = int(width / num_parts)
    part_height = int(height / num_parts)

    with open(filename, "w") as f:
        for i in range(num_parts):
            for j in range(num_parts):
                new_x1 = x1 + i * part_width
                new_y1 = y1 + j * part_height
                new_x2 = new_x1 + part_width
                new_y2 = new_y1 + part_height

                f.write(f"{BatchInferenceCommand} --coordinates {new_x1},{new_y1},{new_x2},{new_y2}\n")

def split_polygon_into_segments(gpkg_path: str, output: str, number_of_split: int):
    '''
    Takes an input path to a .gpkg file.
    Takes an output path to the output file ex. my_path/output.gpkg
    Takes an integer of number of split. It will split each side, i.e. it will result in number_of_split*number_of_split number of tiles
    '''
    gpdf = gp.read_gpkg(gpkg_path)

    big_polygon = gpdf.geometry.unary_union
    bbox = big_polygon.bounds
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    split_width = width / number_of_split
    split_height = height / number_of_split
    polygons = []
    coords_list = []
    for i in range(number_of_split):
        for j in range(number_of_split):
            minx = bbox[0] + i * split_width
            miny = bbox[1] + j * split_height
            maxx = minx + split_width
            maxy = miny + split_height
            poly_bbox = box(minx, miny, maxx, maxy)
            poly = poly_bbox.intersection(big_polygon)
            if poly.is_empty:
                continue
            polygons.append(poly)
            coords = [int(minx), int(miny), int(maxx), int(maxy)]
            coords_list.append(coords)

    new_gpdf = geopandas.GeoDataFrame(geometry=polygons)

    gp.save_bounding_boxes(new_gpdf, output, layer='BoundingBoxes')
    return coords_list
    

def generate_from_gpkg(gpkg_path: str, output: str, number_of_split: int, BatchInferenceCommand: str):
    '''
    Takes an input path to a .gpkg file.
    Takes an output path to the output file ex. "my_path/output.gpkg"
    Takes an integer of number of split. It will split each side, i.e. it will result in number_of_split*number_of_split number of tiles
    Takes a start command to append the coordinates on. ex. python "BatchInference --weight my_weight_file.pt"
    '''
    coords = split_polygon_into_segments(gpkg_path, output, number_of_split)
    with open(output.replace('.gpkg','.sh'), "w") as f:
        for rect in coords:
            
            f.write(f"{BatchInferenceCommand} --coordinates {rect[0]},{rect[1]},{rect[2]},{rect[3]}\n")

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpkg', action='store', type=str)
    parser.add_argument('--output', action='store', type=str)
    parser.add_argument('--number_of_split', action='store', type=int)
    parser.add_argument('--command', action='store', type=str)
    opt = parser.parse_args()

    generate_from_gpkg(opt.gpkg, opt.output, opt.number_of_split, opt.command)