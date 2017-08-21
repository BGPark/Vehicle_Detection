import numpy as np
import itertools

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils import *

cars, notcars, info = get_data_and_info('data/**/*.jpeg')


color_space = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
orient = [6, 9, 12, 15]
pix_per_cell = [6, 8, 10]  # HOG pixels per cell
cell_per_block = [2, 3]  # HOG cells per block
hog_channel = [0, 1, 2, 'ALL']  # Can be 0, 1, 2, or "ALL"
spatial_size = [(16, 16), (16, 32), (32, 16), (32, 32)]  # Spatial binning dimensions
hist_bins = [16, 32]  # Number of histogram bins
spatial_feat = [False, True]  # Spatial features on or off
hist_feat = [False, True]  # Histogram features on or off
hog_feat = [False, True]  # HOG features on or off

count = 0
top_accuracy = 0

for set in itertools.product(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat):
    color_space = set[0]
    orient = set[1]
    pix_per_cell = set[2]
    cell_per_block = set[3]
    hog_channel = set[4]
    spatial_size = set[5]
    hist_bins = set[6]
    spatial_feat = set[7]
    hist_feat = set[8]
    hog_feat = set[9]

    # print('try %d:  %s' % (count, set))
    if not (spatial_feat or hist_feat or hog_feat ):
        continue
    count += 1

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)

    svc = LinearSVC()
    # Check the training time for the SVC
    svc.fit(X_train, y_train)

    accuracy = svc.score(X_test, y_test)
    if(top_accuracy < accuracy):
        top_accuracy = accuracy
        print('%f:  %s' % (top_accuracy, set))







