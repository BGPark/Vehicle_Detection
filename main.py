from utils import *
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
import pickle
from scipy.ndimage.measurements import label
from collections import deque

# Load trained model
dist_pickle = pickle.load(open("model2_save.pkl", "rb"))
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


def proc_pipe(img):
    if proc_pipe.frame_count % PER_FRAMES is 0:
        heat = np.zeros_like(img[:, :, 0]).astype(np.float)

        xstart = 16
        xstop = 1280
        ystart = 370
        ystop = 700
        scale = 2

        windows = find_cars(img, xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                            spatial_size, hist_bins)
        heat = add_heat(heat, windows)

        xstart = 0
        xstop = 1280
        ystart = 380
        ystop = 600
        scale = 1.5
        windows = find_cars(img, xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                            spatial_size, hist_bins)
        heat = add_heat(heat, windows)

        xstart = 640 - 200
        xstop = 640 + 200
        ystart = 390
        ystop = 500
        scale = 1
        windows = find_cars(img, xstart, xstop, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block,
                            spatial_size, hist_bins)
        heat = add_heat(heat, windows)

        # queue whole windows
        proc_pipe.heats.append(heat)

    proc_pipe.frame_count += 1
    heat_mean = np.sum(proc_pipe.heats, axis=0)

    # For video output, threshold doesn't apply here
    heat_image = np.zeros_like(img)
    heat_image[:, :, 0] = heat_mean
    heat_image = heat_image * 20

    # Apply threshold to help remove false positives
    # heat_mean = apply_threshold(heat_mean, len(proc_pipe.heats)*2.5)
    heat_mean = apply_threshold(heat_mean, 4)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat_mean, 0, 255)

    labels = label(heatmap)

    # draw_img = draw_boxes(img, windows)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    compose = np.zeros((720, 1280*2, 3), dtype=np.uint8)
    compose[0:720, 0:1280] = cv2.resize(draw_img, (1280, 720))
    compose[0:720, 1280:1280*2] = cv2.resize(heat_image, (1280, 720))

    return compose


MAX_HEAT_COUNT = 2
proc_pipe.heats = deque(maxlen=MAX_HEAT_COUNT)

PER_FRAMES = 3
proc_pipe.frame_count = 0


if __name__ == '__main__':
    video_file = 'project_video.mp4'
    # video_file = 'IMG_2650.mp4'
    video_output = 'output_images/' + video_file
    # read_clip = VideoFileClip(video_file, audio=False).subclip(38, 42)
    read_clip = VideoFileClip(video_file, audio=False)
    white_clip = read_clip.fl_image(proc_pipe)



    white_clip.write_videofile(video_output, audio=False)
