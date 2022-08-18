import os
import shutil

def main(bounding_boxes, topographical_indice, output_topo_dir):
    for f in os.listdir(bounding_boxes):
        if f.endswith('.txt'):
            shutil.copy(os.path.join(topographical_indice, f.replace('.txt', '.tif')), os.path.join(output_topo_dir, f.replace('.txt', '.tif')))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Merge bounding boxes '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('bounding_boxes', help= 'path to directory with merged bounding boxes')
    parser.add_argument('topographical_indice', help= 'Path to dir of image chips from topographical incides')
    parser.add_argument('output_topo_dir', help= 'path do directory where selected topographical data will be stored')

    args = vars(parser.parse_args())
    main(**args)