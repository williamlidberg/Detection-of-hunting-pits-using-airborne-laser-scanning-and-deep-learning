import os
import random
import numpy as np


def main(input_path, output_csv):
    labels = os.listdir(input_path)
    number_of_test_files = int(len(labels)*0.2)
    test_files = random.sample(labels, number_of_test_files)
    np.savetxt(output_csv, test_files, delimiter=", ", fmt="% s")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Randomly samples 20 percent of the data to be used as test data. The sampled filenames are stored in a csv',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_path', help='path to directory of original_images')
    parser.add_argument('output_csv', help='path to csv')

    args = vars(parser.parse_args())
    main(**args)