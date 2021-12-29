import os
import tifffile as tif
import numpy as np


labels_32bit = 'D:/temp/labels/'
labels_8bit = 'D:/temp/labels8bit/'

os.chmod(labels_8bit, 755)

for file in os.listdir(labels_32bit):
    label32 = labels_32bit + file
    image = tif.imread(label32)
    
    img8bit = image.astype(np.uint8)
    label8bit = labels_8bit + file

    tif.imwrite(labels_8bit, img8bit)