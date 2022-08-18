# adapted from https://blog.paperspace.com/train-yolov5-custom-data/
import os
from sklearn.model_selection import train_test_split

def copy_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False


def main(original_images_dir, original_annotations_dir, train_images_dir, val_images_dir, test_images_dir, train_annotations_dir, val_annotation_dir, test_annotations_dir):
    images = [os.path.join(original_images_dir, x) for x in os.listdir(original_images_dir) if x.endswith('.tif')]
    annotations = [os.path.join(original_annotations_dir, x) for x in os.listdir(original_annotations_dir) if x[-3:] == "txt"]

    images.sort()
    annotations.sort()

    train_images, val_images, train_annotations, val_annotations = train_test_split(images, annotations, test_size = 0.2, random_state = 1)
    val_images, test_images, val_annotations, test_annotations = train_test_split(val_images, val_annotations, test_size = 0.5, random_state = 1)

    # copy the splits into their folders
    copy_files_to_folder(train_images, train_images_dir)
    copy_files_to_folder(val_images, val_images_dir)
    copy_files_to_folder(test_images, test_images_dir)
    copy_files_to_folder(train_annotations, train_annotations_dir)
    copy_files_to_folder(val_annotations, val_annotation_dir)
    copy_files_to_folder(test_annotations, test_annotations_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='split input raster tiles into smaller chips '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('original_images_dir', help='path to directory of original_images')
    parser.add_argument('original_annotations_dir', help = 'path to directory of original_annotations')
    parser.add_argument('train_images_dir', help='path to directory of train_images')
    parser.add_argument('val_images_dir', help = 'path to directory of val_images')
    parser.add_argument('test_images_dir', help='path to directory of test_images')
    parser.add_argument('train_annotations_dir', help = 'path to directory of train_annotations')
    parser.add_argument('val_annotation_dir', help='path to directory of val_annotation')
    parser.add_argument('test_annotations_dir', help = 'path to directory of test_annotations')
    args = vars(parser.parse_args())
    main(**args)
