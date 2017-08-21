import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

from utils import *

# image = mpimg.imread('cutouts/bbox-example-image.jpg')


image = mpimg.imread('project_video[00 00 21].bmp')

params = {}

window_list = None


x_start_stop = [16, None]
y_start_stop = [370, 700]
xy_window = (192, 192)
xy_overlap = (0.5, 0.5)
draw_color = (0, 0, 255)
windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                       xy_window=xy_window, xy_overlap=xy_overlap)

window_list = np.array(windows)

print('window count = %d' % len(windows))

window_img = draw_boxes(image, windows, color=draw_color, thick=6)


x_start_stop = [None, None]
y_start_stop = [390, 600]
xy_window = (128, 128)
xy_overlap = (0.5, 0.5)
draw_color = (0, 255, 0)

windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                       xy_window=xy_window, xy_overlap=xy_overlap)
window_list = np.vstack((window_list, windows))
print('window count = %d' % len(windows))

window_img = draw_boxes(window_img, windows, color=draw_color, thick=6)


x_start_stop = [None, None]
y_start_stop = [410, 510]
xy_window = (64, 64)
xy_overlap = (0.5, 0.5)
draw_color = (255, 0, 0)

windows = slide_window(image, x_start_stop=x_start_stop, y_start_stop=y_start_stop,
                       xy_window=xy_window, xy_overlap=xy_overlap)
window_list = np.vstack((window_list, windows))
print('window count = %d' % len(windows))

window_img = draw_boxes(window_img, windows, color=draw_color, thick=6)
print('total window count = %d' % len(window_list))

test = {}

test["windows"] = window_list

pickle.dump(test, open('windows.p', 'wb'))


test2 = pickle.load(open('windows.p', 'rb'))
print(test2['windows'].shape)

plt.imshow(window_img)
plt.show()


