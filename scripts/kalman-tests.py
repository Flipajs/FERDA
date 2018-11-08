from __future__ import print_function
from __future__ import unicode_literals
from pykalman import KalmanFilter
from core.project.project import Project
from core.graph.region_chunk import RegionChunk
import numpy as np

p = Project()
p.load('/Users/flipajs/Documents/wd/FERDA/rep1-cam3')

rch = RegionChunk(p.chm[1], p.gm, p.rm)
pos = []
for r in rch.regions_gen():
    pos.append(r.centroid())

pos = np.array(pos)
print(pos.shape)

kf = KalmanFilter(transition_matrices = [[1, 1], [0, 1]], observation_matrices = [[0.1, 0.5], [-0.3, 0.0]])
measurements = np.asarray([[1,0], [0,0], [0,1]])  # 3 observations
kf = kf.em(measurements, n_iter=5)
(filtered_state_means, filtered_state_covariances) = kf.filter(measurements)
(smoothed_state_means, smoothed_state_covariances) = kf.smooth(measurements)

print(filtered_state_means, filtered_state_covariances)
print(smoothed_state_means, smoothed_state_covariances)
