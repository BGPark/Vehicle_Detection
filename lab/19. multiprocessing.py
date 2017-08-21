from multiprocessing import Pool, Lock, Manager
import itertools
import time
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pickle
import time

from utils import *


def work(params):
    color_space = params[0]
    orient = params[1]
    pix_per_cell = params[2]
    cell_per_block = params[3]
    hog_channel = params[4]
    spatial_size = params[5]
    hist_bins = params[6]
    spatial_feat = params[7]
    hist_feat = params[8]
    hog_feat = params[9]

    if not (spatial_feat or hist_feat or hog_feat):
        return

    dataset = pickle.load(open('full_data.pkl', 'rb'))
    # dataset = pickle.load(open('simple_data.pkl', 'rb'))


    cars = dataset['car']
    notcars = dataset['notcar']

    car_features = extract_features(cars, color_space=color_space,
                                    spatial_size=spatial_size, hist_bins=hist_bins,
                                    orient=orient, pix_per_cell=pix_per_cell,
                                    cell_per_block=cell_per_block,
                                    hog_channel=hog_channel, spatial_feat=spatial_feat,
                                    hist_feat=hist_feat, hog_feat=hog_feat, isURI=False)
    notcar_features = extract_features(notcars, color_space=color_space,
                                       spatial_size=spatial_size, hist_bins=hist_bins,
                                       orient=orient, pix_per_cell=pix_per_cell,
                                       cell_per_block=cell_per_block,
                                       hog_channel=hog_channel, spatial_feat=spatial_feat,
                                       hist_feat=hist_feat, hog_feat=hog_feat, isURI=False)

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

    acc = svc.score(X_test, y_test)
    # lock.acquire()
    # if accuracy.value <= acc:
    accuracy.value = acc
    print('%s: %f:  %s' % (time.time(), accuracy.value, params))
    # total_count.value += 1
    # lock.release()

    return

# Full features
# color_space = ['RGB', 'HSV', 'LUV', 'HLS', 'YUV', 'YCrCb']
# orient = [6, 9, 12, 15]
# pix_per_cell = [6, 8, 10]  # HOG pixels per cell
# cell_per_block = [2, 3]  # HOG cells per block
# hog_channel = [0, 1, 2, 'ALL']  # Can be 0, 1, 2, or "ALL"
# spatial_size = [(16, 16), (16, 32), (32, 16), (32, 32)]  # Spatial binning dimensions
# hist_bins = [16, 32]  # Number of histogram bins
# spatial_feat = [False, True]  # Spatial features on or off
# hist_feat = [False, True]  # Histogram features on or off
# hog_feat = [False, True]  # HOG features on or off

# selected by simple dataset histogram
color_space = ['YCrCb']
orient = [12]
pix_per_cell = [6, 8]  # HOG pixels per cell
cell_per_block = [2]  # HOG cells per block
hog_channel = ['ALL']  # Can be 0, 1, 2, or "ALL"
spatial_size = [(16, 16), (16, 32), (32, 16), (32, 32)]  # Spatial binning dimensions
hist_bins = [32]  # Number of histogram bins
spatial_feat = [True]  # Spatial features on or off
hist_feat = [True]  # Histogram features on or off
hog_feat = [True]  # HOG features on or off


from multiprocessing import Pool


def init(l, acc, count):
    global lock, accuracy, total_count
    lock = l
    accuracy = acc
    total_count = count


if __name__ == '__main__':
    m = Manager()
    lock = m.Lock()
    accuracy = m.Value('acc', 0)
    total_count = m.Value('count', 0)
    print(time.time())
    pool = Pool(initializer=init, initargs=(lock, accuracy, total_count, ), processes=2)
    result = pool.map(work, [item for item in itertools.product(color_space, orient, pix_per_cell, cell_per_block, hog_channel, spatial_size, hist_bins, spatial_feat, hist_feat, hog_feat)])
    # print(accuracy, total_count)


