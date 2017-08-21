import time

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pickle
import cv2
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label

from utils import *


dist_pickle = pickle.load(open("model_save.pkl", "rb"))
svc = dist_pickle["svc"]
X_scaler = dist_pickle["X_scaler"]
orient = dist_pickle["orient"]
pix_per_cell = dist_pickle["pix_per_cell"]
cell_per_block = dist_pickle["cell_per_block"]
spatial_size = dist_pickle["spatial_size"]
hist_bins = dist_pickle["hist_bins"]
color_space = dist_pickle['color_space']
hog_channel = dist_pickle['hog_channel']
spatial_feat = dist_pickle['spatial_feat']
hist_feat = dist_pickle['hist_feat']
hog_feat = dist_pickle['hog_feat']

# img = mpimg.imread('test1.jpg')
img = mpimg.imread('project_video[00 00 10].bmp')




heat = np.zeros_like(img[:, :, 0]).astype(np.float)


xstart = 16
xstop = 1280
ystart = 370
ystop = 700
scale = 2

# t1 = time.time()
# out_img = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
windows = find_cars(img, xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
heat = add_heat(heat, windows)

# t2 = time.time()
# print(t2-t1)
xstart = 0
xstop = 1280
ystart = 390
ystop = 590
scale = 1.5
# t1 = time.time()
# out_img = find_cars(out_img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
windows = find_cars(img, xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
heat = add_heat(heat, windows)
#
xstart = 0
xstop = 1280
ystart = 410
ystop = 510
scale = 1
# t1 = time.time()
# out_img = find_cars(out_img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
windows = find_cars(img, xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
heat = add_heat(heat, windows)


# Apply threshold to help remove false positives
heat = apply_threshold(heat, 1)

# Visualize the heatmap when displaying
heatmap = np.clip(heat, 0, 255)

labels = label(heatmap)
# draw_img = draw_boxes(img, windows)
draw_img = draw_labeled_bboxes(np.copy(img), labels)


fig = plt.figure()
plt.subplot(121)
plt.imshow(draw_img)
plt.title('Car Positions')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
fig.tight_layout()
plt.show()

print('hold debug')
