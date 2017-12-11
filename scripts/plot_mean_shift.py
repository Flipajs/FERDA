import math

print(__doc__)

import numpy as np
import cv2
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs

###############################################################################
# Generate sample data
# centers = [[1, 1], [-1, -1], [1, -1]]
# X, _ = make_blobs(n_samples=10000, centers=centers, cluster_std=0.6)

# im = cv2.imread('/home/flipajs/Pictures/test/im_014.png')
im = cv2.imread('/home/flipajs/Pictures/test/ant2.png')
h, w, _ = im.shape

X = np.zeros((h*w, 4), dtype=np.uint8)
for i in range(h):
    for j in range(w):
        X[i*w+j, :] = [im[i,j,0], im[i,j,1], im[i,j,2], math.sqrt(i*i + j*j)]


print X.shape
###############################################################################
# Compute prepare_region_cardinality_samples with MeanShift

# The following bandwidth can be automatically detected using
bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=5000)

# ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

labels_unique = np.unique(labels)
n_clusters_ = len(labels_unique)

print("number of estimated clusters : %d" % n_clusters_)

###############################################################################
# Plot result
import pylab as pl
from itertools import cycle

pl.figure(1)
pl.clf()

print labels.shape
limg = np.reshape(labels, (h,w))
pl.matshow(limg)

# colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
# for k, col in zip(range(n_clusters_), colors):
#     my_members = labels == k
#     cluster_center = cluster_centers[k]
#     pl.plot(X[my_members, 0], X[my_members, 1], col + '.')
#     pl.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
#             markeredgecolor='k', markersize=14)
# pl.title('Estimated number of clusters: %d' % n_clusters_)
pl.show()