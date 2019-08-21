# import the necessary packages
import numpy as np
import cv2
import os
from os import listdir
from PIL import Image

input_dir = 'D:/AI/tests/cut/'
output_dir = 'D:/AI/tests/edges/'

def auto_canny(image, sigma=0.33):
    # compute the median of the single channel pixel intensities
    v = np.median(image)

    # apply automatic Canny edge detection using the computed median
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged


for filename in listdir(input_dir):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        filename_png = os.path.splitext(filename)[0] + '.png'
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename_png)
        if os.path.isfile(output_path): continue
        try:
            print('Processing: ' + filename)
            image = cv2.imread(input_dir+filename)
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            blurred = cv2.GaussianBlur(gray, (3, 3), 0)
            auto = auto_canny(blurred)
            auto = cv2.bitwise_not(auto)
            img = Image.fromarray(auto)
            img.save(output_dir+filename_png)
        except Exception as e:
            print('Error: ' + str(e))