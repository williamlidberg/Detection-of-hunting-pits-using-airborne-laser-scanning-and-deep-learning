import csv
import shutil
import re

def main(original_images, moved_test_images, test_chips_csv):
    with open(test_chips_csv, newline='') as f:
        reader = csv.reader(f, delimiter=',')
        test_list = list(reader)
        for chip in test_list:
            name = str(chip)
            cleaned_name = name.replace("[","").replace("]", "").replace("'","")
            original_chips = original_images + cleaned_name
            moved_chips = moved_test_images + cleaned_name
            shutil.move(original_chips, moved_chips)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='moves test chips to a new directory using previsouly created partition csv',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('original_images', help='path to directory of original_images')
    parser.add_argument('moved_test_images', help='path to directory of moved_test_images')
    parser.add_argument('test_chips_csv', help='path to directory of test_chips_csv')


    args = vars(parser.parse_args())
    main(**args)