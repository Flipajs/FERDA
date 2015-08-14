__author__ = 'flipajs'

import cv2
from scripts.region_graph3 import NodeGraphVisualizer, visualize_nodes
from core.region.region import Region
import numpy as np


# path to some testing image...d
im = cv2.imread('/Users/flipajs/Desktop/red_vid.png')

r = Region()
r.pts_ = np.asarray(np.random.rand(500, 2) * 50 + 100, dtype=np.uint8)


vis = visualize_nodes(im, r, margin=1.0)
cv2.imshow('vis', vis)
cv2.waitKey(0)

