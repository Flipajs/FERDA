import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage import data, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.filter import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte


# Load picture and detect edges
# image = img_as_ubyte(data.coins()[0:95, 70:370])

im = cv2.imread('/home/flipajs/Pictures/test/ant2.png')
image = im[:,:,1]

edges = canny(image, sigma=2, low_threshold=5, high_threshold=30)

ed = np.asarray(edges, dtype=np.uint8)
cv2.imshow("edges", ed*255)

fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(5, 2))

# Detect two radii
hough_radii = np.arange(15, 30, 2)
hough_res = hough_circle(edges, hough_radii)

centers = []
accums = []
radii = []

for radius, h in zip(hough_radii, hough_res):
    # For each radius, extract two circles
    num_peaks = 2
    peaks = peak_local_max(h, num_peaks=num_peaks)
    centers.extend(peaks)
    accums.extend(h[peaks[:, 0], peaks[:, 1]])
    radii.extend([radius] * num_peaks)

# Draw the most prominent 5 circles
image = color.gray2rgb(image)
for idx in np.argsort(accums)[::-1][:5]:
    center_x, center_y = centers[idx]
    radius = radii[idx]
    cx, cy = circle_perimeter(center_y, center_x, radius)
    image[cy, cx] = (220, 20, 20)


ax.imshow(image, cmap=plt.cm.gray)
cv2.imshow("im", image)
cv2.waitKey(0)