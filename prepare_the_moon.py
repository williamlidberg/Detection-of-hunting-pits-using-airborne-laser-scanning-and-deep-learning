import geopandas as gpd
import utils.create_labels

def select_and_buffer(all_creaters, buffered_craters):
    raw_data = gpd.read_file(all_creaters)
    select = raw_data[(raw_data['DIAM_C_IM'].between(2.2, 3.8))]
    select['diameter_meter'] = select['DIAM_C_IM']/50 # I use 50 instead of 100 to get a slightly bigger buffer polygon
    select['class'] = 1
    print(len(select))
    # buffer points with field
    buffered_craters = select.buffer(distance=select['diameter_meter'])
    buffered_craters.to_file(all_creaters)


def main(original_craters, buffered_craters, base_file_path, path_to_labeled_moon):
    select_and_buffer(original_craters, buffered_craters)
    utils.create_labels.convert_polygon(base_file_path, buffered_craters, path_to_labeled_moon)

if __name__== '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description='copy chips with labeled pixels',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_creaters', help='Path to shapefile with original craters')
    parser.add_argument('output_creaters', help='path to selected craters')

    args = vars(parser.parse_args())
    main(**args)
