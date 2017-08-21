import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from utils import *

image = mpimg.imread('cutouts/bbox-example-image.jpg')
# image = mpimg.imread('temp-matching-example-2.jpg')
templist = ['cutouts/cutout1.jpg', 'cutouts/cutout2.jpg', 'cutouts/cutout3.jpg',
            'cutouts/cutout4.jpg', 'cutouts/cutout5.jpg', 'cutouts/cutout6.jpg']


bboxes = find_matches(image, templist)
result = draw_boxes(image, bboxes)
plt.imshow(result)
plt.show()