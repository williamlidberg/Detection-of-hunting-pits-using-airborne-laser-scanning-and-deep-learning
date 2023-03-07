import os
import glob
import os.path
import shutil
import time

def merge_files(file_one, file_two, merged_file):
	content = ''
	with open(file_one, 'r') as reader:
		content += reader.read()
	with open(file_two, 'r') as reader:
		content += reader.read()
	with open(merged_file, 'w') as writer: 
		writer.write(content)
	return content


def main(boxdir1, boxdir2, mergeddir):
    files_boxdir1 = os.listdir(boxdir1)
    files_boxdir2 = os.listdir(boxdir2)

    boxdir1_content = ''
    for f in files_boxdir1:
        if f in files_boxdir2:
            boxdir1_content += merge_files(os.path.join(boxdir1, f), os.path.join(boxdir2, f), os.path.join(mergeddir, f))
        else:
            with open(os.path.join(boxdir1, f), 'r') as reader:
                boxdir1_content += reader.read()
        

    boxdir2_content = ''
    for f in files_boxdir2:
        if f in files_boxdir1:
            boxdir2_content += merge_files(os.path.join(boxdir1, f), os.path.join(boxdir2, f), os.path.join(mergeddir, f))
        else:
            with open(os.path.join(boxdir2, f), 'r') as reader:
                boxdir2_content += reader.read()


    #time.sleep(10) # Sleep for 3 seconds
    #print('sleeping for 10 seconds')
    # copy over remaining files from boxdir1 which did not have any overlap between the directories
    for f in os.listdir(boxdir1):
        if f not in os.listdir(mergeddir):
            shutil.copy((os.path.join(boxdir1,f)),(os.path.join(mergeddir, f)))


    # copy over remaining files from boxdir2 which did not have any overlap between the directories
    for f in os.listdir(boxdir2):
        if f not in os.listdir(mergeddir):
            shutil.copy((os.path.join(boxdir2,f)),(os.path.join(mergeddir, f)))

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
                       description='Merge bounding boxes '
                                   'image(s)',
                       formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('boxdir1', help= 'path to directory with csvs containing bounding boxes 1')
    parser.add_argument('boxdir2', help= 'path to directory with csvs containing bounding boxes 2')
    parser.add_argument('mergeddir', help= 'path do directory where merged bounding boxes will be stored')

    args = vars(parser.parse_args())
    main(**args)