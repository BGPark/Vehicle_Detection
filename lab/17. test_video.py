# from cal_camera import get_camera_cal, undistort
from utils import *
import cv2
import matplotlib.pyplot as plt
from scipy import misc
import numpy as np
from moviepy.editor import VideoFileClip
import pickle
from scipy.ndimage.measurements import label

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

image = mpimg.imread('test1.jpg')


windows = pickle.load(open('windows.p', 'rb'))['windows']

def proc_pipe(image):
    hot_windows = search_windows(image, windows, svc, X_scaler, color_space=color_space,
                                 spatial_size=spatial_size, hist_bins=hist_bins,
                                 orient=orient, pix_per_cell=pix_per_cell,
                                 cell_per_block=cell_per_block,
                                 hog_channel=hog_channel, spatial_feat=spatial_feat,
                                 hist_feat=hist_feat, hog_feat=hog_feat)

    heat = np.zeros_like(image[:, :, 0]).astype(np.float)

    # Add heat to each box in box list
    heat = add_heat(heat, hot_windows)

    # Apply threshold to help remove false positives
    heat = apply_threshold(heat, 10)

    # Visualize the heatmap when displaying
    heatmap = np.clip(heat, 0, 255)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    # draw_img = draw_labeled_bboxes(np.copy(image), labels)
    draw_img = draw_boxes(image, hot_windows, (0, 0, 255), thick=3)
    return draw_img


if __name__ == '__main__':
    video_file = 'project_video_40-42.mp4'
    # video_file = 'challenge_video.mp4'
    video_output = 'output_images/' + video_file
    read_clip = VideoFileClip(video_file, audio=False).subclip(38, 40)
    # read_clip = VideoFileClip(video_file, audio=False)
    white_clip = read_clip.fl_image(proc_pipe)
    white_clip.write_videofile(video_output, audio=False)