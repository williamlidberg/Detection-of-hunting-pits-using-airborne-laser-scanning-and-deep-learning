import os
import glob
import shutil

original_dir = '/worksapce/lidar/original/'
out_dir = '/workspace/lidar/pooled_laz_files/'
list_laz_files = glob.glob('/workspace/lidar/original/**/**/*.laz', recursive = True)

non_border = []
for i in list_laz_files:
    if 'border' not in i:
        filename = os.path.basename(i)
        out = out_dir + filename
        shutil.copy(i, out)
        print('copied ', i)