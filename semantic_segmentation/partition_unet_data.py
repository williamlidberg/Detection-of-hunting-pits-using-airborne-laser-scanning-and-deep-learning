# adapted from https://blog.paperspace.com/train-yolov5-custom-data/
from sklearn.model_selection import train_test_split

def copy_files_to_folder(list_of_files, destination_folder):
    for f in list_of_files:
        try:
            shutil.copy(f, destination_folder)
        except:
            print(f)
            assert False


def main(original_images_dir, original_hillshade_dir, original_hpmf_dir,original_slope, original_stdon, original_annotations_dir, train_images_dir, val_images_dir, test_images_dir, train_annotations_dir, val_annotation_dir, test_annotations_dir):
    images = [os.path.join(original_images_dir, x) for x in os.listdir(original_images_dir) if x.endswith('.tif')]
    hillshade = [os.path.join(original_hillshade_dir, x) for x in os.listdir(original_hillshade_dir) if x.endswith('.tif')]
    annotations = [os.path.join(original_annotations_dir, x) for x in os.listdir(original_annotations_dir) if x[-3:] == "txt"]

    images.sort()
    annotations.sort()

    train_hillshade,test_hillshade, train_labels,test_labels = train_test_split(hillshade, annotations, test_size = 0.2, random_state = 1)


    # copy the splits into their folders
    # train
    copy_files_to_folder(train_hillshade, train_images_dir)
    copy_files_to_folder(train_hpmf, train_hpmf_dir)
    copy_files_to_folder(train_slope, train_slope_dir)
    copy_files_to_folder(train_stdon, train_stdon_dir)
    copy_files_to_folder(train_labels, train_label_dir)
    # test
    copy_files_to_folder(test_hillshade, test_hillshade_dir)
    copy_files_to_folder(test_hpmf, test_hpmf_dir)
    copy_files_to_folder(test_slope, test_slope_dir)
    copy_files_to_folder(test_stdon, test_stdon_dir)
    copy_files_to_folder(test_labels, test_labels_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='split input raster tiles into smaller chips '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('original_hillshade_dir', help='path to directory of original_images')
    parser.add_argument('original__hpmf_dir', help='path to directory of original_images')
    parser.add_argument('original_slope_dir', help='path to directory of original_images')
    parser.add_argument('original_stdon_dir', help='path to directory of original_images')
    parser.add_argument('original_labels_dir', help = 'path to directory of original_labels')

    parser.add_argument('train_hillshade_dir', help='path to directory of train_images')
    parser.add_argument('train_hpmf_dir', help='path to directory of train_images')
    parser.add_argument('train_slope_dir', help='path to directory of train_images')
    parser.add_argument('train_stdon_dir', help='path to directory of train_images')
    parser.add_argument('train_label_dir', help='path to directory of train_images')

    parser.add_argument('test_hillshade_dir', help='path to directory of test_images')
    parser.add_argument('test_hpmf_dir', help='path to directory of test_images')
    parser.add_argument('test_slope_dir', help='path to directory of test_images')
    parser.add_argument('test_stdon_dir', help='path to directory of test_images')
    parser.add_argument('test_labels_dir', help = 'path to directory of test_annotations')
    args = vars(parser.parse_args())
    main(**args)