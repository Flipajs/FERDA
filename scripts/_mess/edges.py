__author__ = 'filip@naiser.cz'

import numpy as np
from numpy import random
from matplotlib import pyplot as plt
import cv2, cv

img = cv2.imread("../out/frames/frame2.png", 0)
cv2.imshow("test", img)
edges = cv2.Canny(img, 1, 50)

cv2.imwrite("../out/frames/frame2-edges.png", edges)

plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
