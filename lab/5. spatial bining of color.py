import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from utils import *

image = mpimg.imread('../test_img.jpg')

print(image.ravel().shape)
small_img = cv2.resize(image, (32, 32))
print(small_img.shape)
(32, 32, 3)
print(small_img.ravel().shape)

# Read in an image
# You can also read cutout2, 3, 4 etc. to see other examples
image = mpimg.imread('../cutouts/cutout1.jpg')



feature_vec = bin_spatial(image, color_space='LUV', size=(32, 32))

# Plot features
plt.plot(feature_vec)
plt.title('Spatially Binned Features')
plt.show()