import os
import argparse
import re

# example
# python /workspace/code/tools/drop_strange_boxes.py /workspace/data/final_data_05m/training/ /workspace/data/small_bounding_boxes/final_data_05m_filtered_training_boxes.txt

def main(original_dir, bad_chips_txt):
    with open(bad_chips_txt) as f:
        lines = f.readlines()
    errorboxes = []
    for line in lines:
        if line.startswith("! Bounding box"):
            errorboxes.append(line)

    p_file = re.compile(r'! Bounding box in file (\d+\.txt)')
    chips_to_drop = []
    for i in errorboxes:
        chips_to_drop.append(p_file.findall(i)[0])

    for subdir, dirs, files in os.walk(original_dir):
        for file in files:
            if file in chips_to_drop and file.endswith('.txt'): 
                print('removed ',(os.path.join(subdir, file)))
                os.remove(os.path.join(subdir, file))
            if file.replace('.tif','.txt') in chips_to_drop and file.endswith('.tif'):
                print('removed ',(os.path.join(subdir, file)))
                os.remove(os.path.join(subdir, file))


if __name__== '__main__':
    parser = argparse.ArgumentParser(
        description='Drop chips with bad bounding boxes',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('original_dir', help='Path to directory where laz files are stored')   
    parser.add_argument('bad_chips_txt', help='Path to directory where dem files are stored')  

    args = vars(parser.parse_args())
    main(**args)