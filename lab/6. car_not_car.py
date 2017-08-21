import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob

from utils import *

# from skimage.feature import hog
# from skimage import color, exposure
# images are divided up into vehicles and non-vehicles


cars, notcars, info = get_data_and_info('data/**/*.jpeg', recursive=True)

print('Your function returned a count of',
      info["n_cars"], ' cars and',
      info["n_notcars"], ' non-cars')
print('of size: ', info["image_shape"], ' and data type:',
      info["data_type"])

# Just for fun choose random car / not-car indices and plot example images
car_ind = np.random.randint(0, len(cars))
notcar_ind = np.random.randint(0, len(notcars))

# Read in car / not-car images
car_image = mpimg.imread(cars[car_ind])
notcar_image = mpimg.imread(notcars[notcar_ind])

# Plot the examples
fig = plt.figure()
plt.subplot(121)
plt.imshow(car_image)
plt.title('Example Car Image')
plt.subplot(122)
plt.imshow(notcar_image)
plt.title('Example Not-car Image')
plt.show()