import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from scipy.ndimage.measurements import label

from utils import *

# Read in a pickle file with bboxes saved
# Each item in the "all_bboxes" list will contain a
# list of boxes for one of the images shown above
# box_list = pickle.load(open("bbox_pickle.p", "rb"))
box_list = [((800, 400), (900, 500)), ((850, 400), (950, 500)), ((1050, 400), (1150, 500)), ((1100, 400), (1200, 500)), ((1150, 400), (1250, 500)), ((875, 400), (925, 450)), ((1075, 400), (1125, 450)), ((825, 425), (875, 475)), ((814, 400), (889, 475)), ((851, 400), (926, 475)), ((1073, 400), (1148, 475)), ((1147, 437), (1222, 512)), ((1184, 437), (1259, 512)), ((400, 400), (500, 500))]


# Read in image similar to one shown above
image = mpimg.imread('test1.jpg')
heat = np.zeros_like(image[:, :, 0]).astype(np.float)



# Add heat to each box in box list
heat = add_heat(heat, box_list)

# Apply threshold to help remove false positives
heat = apply_threshold(heat, 1)

# Visualize the heatmap when displaying
heatmap = np.clip(heat, 0, 255)

# Find final boxes from heatmap using label function
labels = label(heatmap)
draw_img = draw_labeled_bboxes(np.copy(image), labels)

fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
plt.show()

